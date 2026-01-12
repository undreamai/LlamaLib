#import <Foundation/Foundation.h>
#import "ios_http_transport.h"

// ---- StreamDelegate: Handles NSURLSession callbacks ----
@interface StreamDelegate : NSObject <NSURLSessionDataDelegate>
@property(nonatomic, assign) bool *cancel_flag;
@property(nonatomic, assign) CharArrayFnWithContext callback;
@property(nonatomic, assign) void *callbackContext;
@property(nonatomic, strong) NSMutableData *buffer;
@property(nonatomic, strong) dispatch_semaphore_t sema;
@property(nonatomic, assign) int *status_code;
@property(nonatomic, strong) NSString *error_message;
@end

@implementation StreamDelegate

- (instancetype)initWithCallback:(CharArrayFnWithContext)callback
                  callbackContext:(void*)callbackContext
                       cancelFlag:(bool*)cancel_flag
                       statusCode:(int*)status_code
                             sema:(dispatch_semaphore_t)sema
{
    self = [super init];
    if (self) {
        self.callback = callback;
        self.callbackContext = callbackContext;
        self.cancel_flag = cancel_flag;
        self.status_code = status_code;
        self.buffer = [NSMutableData data];
        self.sema = sema;
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
    // Check cancellation BEFORE processing
    if (self.cancel_flag && *self.cancel_flag) {
        [dataTask cancel];
        return;
    }
    
    if (self.callback) {
        // Streaming mode: Call callback with each chunk as null-terminated string
        NSString *chunk = [[NSString alloc] initWithData:data 
                                                encoding:NSUTF8StringEncoding];
        if (chunk) {
            // Call the callback with null-terminated C string and context
            const char *cString = [chunk UTF8String];
            self.callback(cString, self.callbackContext);
            
            // Check cancellation AFTER callback
            if (self.cancel_flag && *self.cancel_flag) {
                [dataTask cancel];
            }
        }
    } else {
        // Non-streaming mode: Buffer all data
        [self.buffer appendData:data];
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
    std::string last_error;
    
    Impl(const std::string &h, bool ssl, int p)
        : host(h), port(p), use_ssl(ssl), timeout_seconds(60.0) {}
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

std::string IOSHttpTransport::get_last_error() const {
    return pImpl->last_error;
}

HttpResult IOSHttpTransport::post_request(
    const std::string &path,
    const std::string &body,
    const std::vector<std::pair<std::string, std::string>> &headers,
    CharArrayFnWithContext callback,
    void* callback_context,
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
            pImpl->last_error = result.error_message;
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
                                                            callbackContext:callback_context
                                                                 cancelFlag:cancel_flag
                                                                 statusCode:&status_code
                                                                       sema:sema];
        
        // Configure session - USE BACKGROUND QUEUE (NOT MAIN QUEUE!)
        NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
        config.timeoutIntervalForRequest = pImpl->timeout_seconds;
        config.timeoutIntervalForResource = pImpl->timeout_seconds * 2;
        
        // Create background queue for networking
        NSOperationQueue *queue = [[NSOperationQueue alloc] init];
        queue.maxConcurrentOperationCount = 1;
        queue.qualityOfService = NSQualityOfServiceUserInitiated;
        
        // Create session with system TLS (no custom certificate handling)
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
            pImpl->last_error = result.error_message;
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
        
        // Return buffered data if not streaming
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