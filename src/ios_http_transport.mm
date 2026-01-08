#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import "ios_http_transport.h"

// ---- StreamDelegate: Handles NSURLSession callbacks ----
@interface StreamDelegate : NSObject <NSURLSessionDataDelegate>
@property(nonatomic, assign) bool *cancel_flag;
@property(nonatomic, copy) CharArrayFn callback;
@property(nonatomic, strong) NSMutableData *buffer;
@property(nonatomic, strong) dispatch_semaphore_t sema;
@property(nonatomic, assign) int *status_code;
@property(nonatomic, strong) NSString *error_message;
@property(nonatomic, assign) bool validate_cert;
@property(nonatomic, strong) NSData *pinned_cert_data;
@end

@implementation StreamDelegate

- (instancetype)initWithCallback:(CharArrayFn)callback
                      cancelFlag:(bool*)cancel_flag
                      statusCode:(int*)status_code
                           sema:(dispatch_semaphore_t)sema
{
    self = [super init];
    if (self) {
        self.callback = callback;
        self.cancel_flag = cancel_flag;
        self.status_code = status_code;
        self.buffer = [NSMutableData data];
        self.sema = sema;
        self.validate_cert = YES;
        self.error_message = nil;
    }
    return self;
}

// Handle HTTP response (status code)
- (void)URLSession:(NSURLSession *)session
          dataTask:(NSURLSessionDataTask *)dataTask
didReceiveResponse:(NSURLResponse *)response
 completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler
{
    if ([response isKindOfClass:[NSHTTPURLResponse class]]) {
        NSHTTPURLResponse *httpResponse = (NSHTTPURLResponse *)response;
        if (self.status_code) {
            *self.status_code = (int)httpResponse.statusCode;
        }
    }
    completionHandler(NSURLSessionResponseAllow);
}

// Receive streaming data incrementally
- (void)URLSession:(NSURLSession *)session
          dataTask:(NSURLSessionDataTask *)dataTask
    didReceiveData:(NSData *)data
{
    // Check cancellation
    if (self.cancel_flag && *self.cancel_flag) {
        [dataTask cancel];
        return;
    }
    
    if (self.callback) {
        // Stream data to callback
        const char *bytes = (const char *)data.bytes;
        size_t len = data.length;
        bool cont = self.callback(bytes, len);
        if (!cont) {
            [dataTask cancel];
        }
    } else {
        // Buffer data for non-streaming requests
        [self.buffer appendData:data];
    }
}

// SSL Certificate validation with optional pinning
- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
didReceiveChallenge:(NSURLAuthenticationChallenge *)challenge
 completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition, NSURLCredential *))completionHandler
{
    if (!self.validate_cert) {
        // Skip validation if disabled
        NSURLCredential *credential = [NSURLCredential credentialForTrust:challenge.protectionSpace.serverTrust];
        completionHandler(NSURLSessionAuthChallengeUseCredential, credential);
        return;
    }
    
    if ([challenge.protectionSpace.authenticationMethod isEqualToString:NSURLAuthenticationMethodServerTrust]) {
        SecTrustRef serverTrust = challenge.protectionSpace.serverTrust;
        
        // Certificate pinning (if configured)
        if (self.pinned_cert_data) {
            SecCertificateRef pinnedCert = SecCertificateCreateWithData(
                kCFAllocatorDefault,
                (__bridge CFDataRef)self.pinned_cert_data
            );
            
            if (pinnedCert) {
                // Get server certificate
                SecCertificateRef serverCert = SecTrustGetCertificateAtIndex(serverTrust, 0);
                NSData *serverCertData = (__bridge_transfer NSData *)SecCertificateCopyData(serverCert);
                
                // Compare certificates
                if ([serverCertData isEqualToData:self.pinned_cert_data]) {
                    NSURLCredential *credential = [NSURLCredential credentialForTrust:serverTrust];
                    completionHandler(NSURLSessionAuthChallengeUseCredential, credential);
                    CFRelease(pinnedCert);
                    return;
                }
                CFRelease(pinnedCert);
            }
            
            // Pinning failed
            self.error_message = @"Certificate pinning validation failed";
            completionHandler(NSURLSessionAuthChallengeCancelAuthenticationChallenge, nil);
            return;
        }
        
        // Standard system validation
        NSURLCredential *credential = [NSURLCredential credentialForTrust:serverTrust];
        completionHandler(NSURLSessionAuthChallengeUseCredential, credential);
    } else {
        completionHandler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
    }
}

// Completion callback
- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
didCompleteWithError:(NSError *)error
{
    if (error) {
        self.error_message = error.localizedDescription;
        if (self.status_code && *self.status_code == 0) {
            *self.status_code = (int)error.code;
        }
    }
    dispatch_semaphore_signal(self.sema);
}

@end

// ---- IOSHttpTransport::Impl (PIMPL) ----
class IOSHttpTransport::Impl {
public:
    std::string host;
    int port;
    bool use_ssl;
    double timeout_seconds;
    std::string pinned_cert_pem;
    bool validate_certificates;
    std::string last_error;
    
    Impl(const std::string &h, bool ssl, int p)
        : host(h), port(p), use_ssl(ssl), timeout_seconds(60.0),
          validate_certificates(true) {}
};

// ---- IOSHttpTransport Implementation ----
IOSHttpTransport::IOSHttpTransport(const std::string &host, bool use_ssl, int port)
    : pImpl(std::make_unique<Impl>(host, use_ssl, port))
{
}

IOSHttpTransport::~IOSHttpTransport() = default;

void IOSHttpTransport::set_timeout(double timeout_seconds) {
    pImpl->timeout_seconds = timeout_seconds;
}

void IOSHttpTransport::set_certificate_pinning(const std::string &cert_pem) {
    pImpl->pinned_cert_pem = cert_pem;
}

void IOSHttpTransport::enable_certificate_validation(bool enable) {
    pImpl->validate_certificates = enable;
}

bool IOSHttpTransport::is_connected() {
    // Could implement a simple health check here
    return true;
}

std::string IOSHttpTransport::get_last_error() const {
    return pImpl->last_error;
}

HttpResult IOSHttpTransport::post_request(
    const std::string &path,
    const std::string &body,
    const std::vector<std::pair<std::string, std::string>> &headers,
    CharArrayFn callback,
    bool *cancel_flag)
{
    @autoreleasepool {
        HttpResult result;
        
        // Build URL
        NSString *scheme = pImpl->use_ssl ? @"https" : @"http";
        NSString *urlStr;
        if (pImpl->port > 0) {
            urlStr = [NSString stringWithFormat:@"%@://%s:%d/%s",
                     scheme, pImpl->host.c_str(), pImpl->port, path.c_str()];
        } else {
            urlStr = [NSString stringWithFormat:@"%@://%s/%s",
                     scheme, pImpl->host.c_str(), path.c_str()];
        }
        
        NSURL *url = [NSURL URLWithString:urlStr];
        if (!url) {
            result.error_message = "Invalid URL: " + std::string([urlStr UTF8String]);
            return result;
        }
        
        // Create request
        NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
        request.HTTPMethod = @"POST";
        request.HTTPBody = [NSData dataWithBytes:body.data() length:body.size()];
        request.timeoutInterval = pImpl->timeout_seconds;
        
        // Set headers
        for (const auto &h : headers) {
            NSString *key = [NSString stringWithUTF8String:h.first.c_str()];
            NSString *value = [NSString stringWithUTF8String:h.second.c_str()];
            [request setValue:value forHTTPHeaderField:key];
        }
        
        // Create semaphore for synchronous operation
        dispatch_semaphore_t sema = dispatch_semaphore_create(0);
        
        int status_code = 0;
        StreamDelegate *delegate = [[StreamDelegate alloc] initWithCallback:callback
                                                                 cancelFlag:cancel_flag
                                                                 statusCode:&status_code
                                                                       sema:sema];
        delegate.validate_cert = pImpl->validate_certificates;
        
        // Configure certificate pinning if provided
        if (!pImpl->pinned_cert_pem.empty()) {
            // Convert PEM to DER
            NSString *pemStr = [NSString stringWithUTF8String:pImpl->pinned_cert_pem.c_str()];
            NSString *base64 = [pemStr stringByReplacingOccurrencesOfString:@"-----BEGIN CERTIFICATE-----" withString:@""];
            base64 = [base64 stringByReplacingOccurrencesOfString:@"-----END CERTIFICATE-----" withString:@""];
            base64 = [base64 stringByReplacingOccurrencesOfString:@"\n" withString:@""];
            base64 = [base64 stringByReplacingOccurrencesOfString:@"\r" withString:@""];
            
            NSData *certData = [[NSData alloc] initWithBase64EncodedString:base64 options:0];
            if (certData) {
                delegate.pinned_cert_data = certData;
            }
        }
        
        // Configure session - USE BACKGROUND QUEUE (NOT MAIN QUEUE!)
        NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
        config.timeoutIntervalForRequest = pImpl->timeout_seconds;
        config.timeoutIntervalForResource = pImpl->timeout_seconds * 2;
        
        // Create background queue for networking
        NSOperationQueue *queue = [[NSOperationQueue alloc] init];
        queue.maxConcurrentOperationCount = 1;
        queue.qualityOfService = NSQualityOfServiceUserInitiated;
        
        NSURLSession *session = [NSURLSession sessionWithConfiguration:config
                                                              delegate:delegate
                                                         delegateQueue:queue];
        
        NSURLSessionDataTask *task = [session dataTaskWithRequest:request];
        [task resume];
        
        // Wait for completion (with timeout)
        dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 
                                               (int64_t)((pImpl->timeout_seconds + 5) * NSEC_PER_SEC));
        long wait_result = dispatch_semaphore_wait(sema, timeout);
        
        if (wait_result != 0) {
            // Timeout occurred
            [task cancel];
            [session invalidateAndCancel];
            result.error_message = "Request timeout";
            return result;
        }
        
        // Clean up session
        [session finishTasksAndInvalidate];
        
        // Process results
        result.status_code = status_code;
        
        if (delegate.error_message) {
            result.error_message = [delegate.error_message UTF8String];
            pImpl->last_error = result.error_message;
        }
        
        if (!callback && delegate.buffer.length > 0) {
            result.body = std::string((const char*)delegate.buffer.bytes, 
                                     delegate.buffer.length);
        }
        
        // Check if request was successful
        result.success = (status_code >= 200 && status_code < 300) && 
                        (delegate.error_message == nil);
        
        if (!result.success && result.error_message.empty()) {
            result.error_message = "HTTP error: " + std::to_string(status_code);
            pImpl->last_error = result.error_message;
        }
        
        return result;
    }
}