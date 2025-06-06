#!/bin/bash

# Universal Curl Build Script with OpenSSL and zlib
# Supports: iOS, visionOS, and Android
# This script builds curl, OpenSSL, and zlib for multiple platforms

set -e

# Configuration
WORK_DIR=${WORK_DIR:-$(pwd)/universal-curl-build}
INSTALL_DIR=${INSTALL_DIR:-$(pwd)/universal-curl-install}

# Android Configuration
ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-$HOME/Android/Sdk/ndk/25.2.9519653}
ANDROID_API_LEVEL=${ANDROID_API_LEVEL:-23}

# iOS Configuration
IOS_MIN_VERSION=${IOS_MIN_VERSION:-14.0}
VISIONOS_MIN_VERSION=${VISIONOS_MIN_VERSION:-1.0}

# Version configurations
OPENSSL_VERSION="3.1.4"
ZLIB_VERSION="1.3.1"
CURL_VERSION="8.14.1"

# Platform configurations
PLATFORMS=()
ANDROID_ARCHS=("arm64-v8a" "x86_64")
IOS_ARCHS=("arm64" "x86_64")  # arm64 for device, x86_64 for simulator
VISIONOS_ARCHS=("arm64" "x86_64")  # arm64 for device, x86_64 for simulator

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_platform() {
    echo -e "${BLUE}[PLATFORM]${NC} $1"
}

# Parse command line arguments
parse_arguments() {
    PLATFORMS=()
    
    if [ $# -eq 0 ]; then
        log_info "No platforms specified. Use --help for options."
        show_help
        exit 0
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --android)
                PLATFORMS+=("android")
                shift
                ;;
            --ios)
                PLATFORMS+=("ios")
                shift
                ;;
            --visionos)
                PLATFORMS+=("visionos")
                shift
                ;;
            --all)
                PLATFORMS=("android" "ios" "visionos")
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "Universal Curl Build Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --android      Build for Android (arm64-v8a, armeabi-v7a, x86, x86_64)"
    echo "  --ios          Build for iOS (arm64 device, x86_64 simulator)"
    echo "  --visionos     Build for visionOS (arm64 device, x86_64 simulator)"
    echo "  --all          Build for all platforms"
    echo "  --help         Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ANDROID_NDK_ROOT      Path to Android ANDROID_NDK_ROOT (for Android builds)"
    echo "  ANDROID_API_LEVEL     Android API level (default: 21)"
    echo "  IOS_MIN_VERSION       iOS minimum version (default: 12.0)"
    echo "  VISIONOS_MIN_VERSION  visionOS minimum version (default: 1.0)"
    echo "  WORK_DIR              Build directory (default: ./universal-curl-build)"
    echo "  INSTALL_DIR           Install directory (default: ./universal-curl-install)"
    echo ""
    echo "Examples:"
    echo "  $0 --ios --android    # Build for iOS and Android"
    echo "  $0 --all              # Build for all platforms"
    echo "  $0 --visionos         # Build for visionOS only"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check common tools
    for cmd in wget tar make; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is required but not installed"
            exit 1
        fi
    done
    
    # Platform-specific checks
    for platform in "${PLATFORMS[@]}"; do
        case $platform in
            android)
                if [ ! -d "$ANDROID_NDK_ROOT" ]; then
                    log_error "Android ANDROID_NDK_ROOT not found at: $ANDROID_NDK_ROOT"
                    log_error "Please set ANDROID_NDK_ROOT environment variable"
                    exit 1
                fi
                ;;
            ios|visionos)
                if ! command -v xcodebuild &> /dev/null; then
                    log_error "Xcode and command line tools required for iOS/visionOS builds"
                    exit 1
                fi
                ;;
        esac
    done
    
    log_info "Prerequisites check passed"
}

# Download source files
download_sources() {
    log_info "Downloading source files..."
    
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    # Download OpenSSL
    if [ ! -f "openssl-${OPENSSL_VERSION}.tar.gz" ]; then
        log_info "Downloading OpenSSL ${OPENSSL_VERSION}..."
        wget "https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz"
    fi
    
    # Download zlib
    if [ ! -f "zlib-${ZLIB_VERSION}.tar.gz" ]; then
        log_info "Downloading zlib ${ZLIB_VERSION}..."
        wget "https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz"
    fi
    
    # Download curl
    if [ ! -f "curl-${CURL_VERSION}.tar.gz" ]; then
        log_info "Downloading curl ${CURL_VERSION}..."
        wget "https://curl.se/download/curl-${CURL_VERSION}.tar.gz"
    fi
    
    log_info "Source files downloaded"
}

# Extract source files
extract_sources() {
    log_info "Extracting source files..."
    
    cd "$WORK_DIR"
    
    for archive in openssl-${OPENSSL_VERSION}.tar.gz zlib-${ZLIB_VERSION}.tar.gz curl-${CURL_VERSION}.tar.gz; do
        if [ -f "$archive" ]; then
            tar -xzf "$archive"
        fi
    done
    
    log_info "Source files extracted"
}

# Get Android toolchain info
get_android_toolchain() {
    local arch=$1
    local api_level=$2
    
    case $arch in
        arm64-v8a)
            export ANDROID_ARCH="aarch64"
            export ANDROID_EABI="aarch64-linux-android"
            export OPENSSL_ARCH="android-arm64"
            ;;
        armeabi-v7a)
            export ANDROID_ARCH="armv7a"
            export ANDROID_EABI="armv7a-linux-androideabi"
            export OPENSSL_ARCH="android-arm"
            ;;
        x86)
            export ANDROID_ARCH="i686"
            export ANDROID_EABI="i686-linux-android"
            export OPENSSL_ARCH="android-x86"
            ;;
        x86_64)
            export ANDROID_ARCH="x86_64"
            export ANDROID_EABI="x86_64-linux-android"
            export OPENSSL_ARCH="android-x86_64"
            ;;
        *)
            log_error "Unsupported Android architecture: $arch"
            exit 1
            ;;
    esac
    
    export TOOLCHAIN_ROOT="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64"
    export CC="$TOOLCHAIN_ROOT/bin/${ANDROID_EABI}${api_level}-clang"
    export CXX="$TOOLCHAIN_ROOT/bin/${ANDROID_EABI}${api_level}-clang++"
    export AR="$TOOLCHAIN_ROOT/bin/llvm-ar"
    export RANLIB="$TOOLCHAIN_ROOT/bin/llvm-ranlib"
    export STRIP="$TOOLCHAIN_ROOT/bin/llvm-strip"
    export NM="$TOOLCHAIN_ROOT/bin/llvm-nm"
    
    export CFLAGS="-fPIC -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -static"
    export CXXFLAGS="$CFLAGS -frtti -fexceptions"
    export LDFLAGS="-Wl,--gc-sections -static"
}

# Get iOS/visionOS toolchain info
get_apple_toolchain() {
    local platform=$1
    local arch=$2
    
    case $platform in
        ios)
            if [ "$arch" = "x86_64" ]; then
                export APPLE_SDK="iphonesimulator"
                export APPLE_PLATFORM="iPhoneSimulator"
                export OPENSSL_ARCH="ios-sim-cross-x86_64"
            else
                export APPLE_SDK="iphoneos"
                export APPLE_PLATFORM="iPhoneOS"
                export OPENSSL_ARCH="ios64-cross-arm64"
            fi
            export MIN_VERSION="$IOS_MIN_VERSION"
            ;;
        visionos)
            if [ "$arch" = "x86_64" ]; then
                export APPLE_SDK="xrsimulator"
                export APPLE_PLATFORM="XRSimulator"
                export OPENSSL_ARCH="ios-sim-cross-x86_64"  # Use iOS sim config for visionOS sim
            else
                export APPLE_SDK="xros"
                export APPLE_PLATFORM="XROS"
                export OPENSSL_ARCH="ios64-cross-arm64"  # Use iOS config for visionOS device
            fi
            export MIN_VERSION="$VISIONOS_MIN_VERSION"
            ;;
    esac
    
    export DEVELOPER_DIR=$(xcode-select -print-path)
    export SDK_ROOT="$DEVELOPER_DIR/Platforms/${APPLE_PLATFORM}.platform/Developer/SDKs/${APPLE_SDK}.sdk"
    
    if [ ! -d "$SDK_ROOT" ]; then
        log_error "SDK not found: $SDK_ROOT"
        log_error "Make sure Xcode and the required SDKs are installed"
        exit 1
    fi
    
    export CC="$(xcrun -sdk $APPLE_SDK -find clang)"
    export CXX="$(xcrun -sdk $APPLE_SDK -find clang++)"
    export AR="$(xcrun -sdk $APPLE_SDK -find ar)"
    export RANLIB="$(xcrun -sdk $APPLE_SDK -find ranlib)"
    export STRIP="$(xcrun -sdk $APPLE_SDK -find strip)"
    export NM="$(xcrun -sdk $APPLE_SDK -find nm)"
    
    case $platform in
        ios)
            export CFLAGS="-arch $arch -isysroot $SDK_ROOT -mios-version-min=$MIN_VERSION -fembed-bitcode"
            ;;
        visionos)
            export CFLAGS="-arch $arch -isysroot $SDK_ROOT -mxros-version-min=$MIN_VERSION"
            ;;
    esac
    
    export CXXFLAGS="$CFLAGS"
    export LDFLAGS="$CFLAGS"
}

# Build OpenSSL for Android
build_openssl_android() {
    local arch=$1
    local install_prefix="$INSTALL_DIR/android/$arch"
    
    log_info "Building OpenSSL for Android $arch..."
    
    cd "$WORK_DIR/openssl-${OPENSSL_VERSION}"
    make clean &> /dev/null || true
    
    get_android_toolchain $arch $ANDROID_API_LEVEL
    
    export ANDROID_NDK_ROOT
    export PATH="$TOOLCHAIN_ROOT/bin:$PATH"
    
    ./Configure $OPENSSL_ARCH \
        -D__ANDROID_API__=$ANDROID_API_LEVEL \
        --prefix="$install_prefix" \
        --openssldir="$install_prefix/ssl" \
        no-shared \
        no-dso \
        no-engine \
        no-tests \
        -static
    
    make -j$(nproc)
    make install_sw
    
    log_info "OpenSSL built successfully for Android $arch"
}

# Build OpenSSL for Apple platforms
build_openssl_apple() {
    local platform=$1
    local arch=$2
    local install_prefix="$INSTALL_DIR/$platform/$arch"
    
    log_info "Building OpenSSL for $platform $arch..."
    
    cd "$WORK_DIR/openssl-${OPENSSL_VERSION}"
    make clean &> /dev/null || true
    
    get_apple_toolchain $platform $arch
    
    ./Configure $OPENSSL_ARCH \
        --prefix="$install_prefix" \
        --openssldir="$install_prefix/ssl" \
        no-shared \
        no-dso \
        no-engine \
        no-tests \
        -static
    
    make -j$(nproc)
    make install_sw
    
    log_info "OpenSSL built successfully for $platform $arch"
}

# Build zlib for Android
build_zlib_android() {
    local arch=$1
    local install_prefix="$INSTALL_DIR/android/$arch"
    
    log_info "Building zlib for Android $arch..."
    
    cd "$WORK_DIR/zlib-${ZLIB_VERSION}"
    make clean &> /dev/null || true
    
    get_android_toolchain $arch $ANDROID_API_LEVEL
    
    export CHOST="$ANDROID_EABI"
    
    ./configure --prefix="$install_prefix" --static
    
    make -j$(nproc)
    make install
    
    log_info "zlib built successfully for Android $arch"
}

# Build zlib for Apple platforms
build_zlib_apple() {
    local platform=$1
    local arch=$2
    local install_prefix="$INSTALL_DIR/$platform/$arch"
    
    log_info "Building zlib for $platform $arch..."
    
    cd "$WORK_DIR/zlib-${ZLIB_VERSION}"
    make clean &> /dev/null || true
    
    get_apple_toolchain $platform $arch
    
    ./configure --prefix="$install_prefix" --static
    
    make -j$(nproc)
    make install
    
    log_info "zlib built successfully for $platform $arch"
}

# Build curl for Android
build_curl_android() {
    local arch=$1
    local install_prefix="$INSTALL_DIR/android/$arch"
    
    log_info "Building curl for Android $arch..."
    
    cd "$WORK_DIR/curl-${CURL_VERSION}"
    make clean &> /dev/null || true
    
    get_android_toolchain $arch $ANDROID_API_LEVEL
    
    export PKG_CONFIG_PATH="$install_prefix/lib/pkgconfig"
    export CPPFLAGS="-I$install_prefix/include"
    export LDFLAGS="$LDFLAGS -L$install_prefix/lib -static"
    export LIBS="-lssl -lcrypto -lz"
    
    ./configure \
        --host="$ANDROID_EABI" \
        --target="$ANDROID_EABI" \
        --prefix="$install_prefix" \
        --with-openssl="$install_prefix" \
        --with-zlib="$install_prefix" \
        --enable-static \
        --disable-shared \
        --disable-verbose \
        --disable-manual \
        --disable-ldap \
        --disable-ldaps \
        --disable-rtsp \
        --disable-dict \
        --disable-telnet \
        --disable-tftp \
        --disable-pop3 \
        --disable-imap \
        --disable-smb \
        --disable-smtp \
        --disable-gopher \
        --disable-sspi \
        --disable-threaded-resolver \
        --without-librtmp \
        --without-libidn2 \
        --without-libpsl \
        --without-nghttp2 \
        --without-brotli \
        --without-zstd \
        --without-libgsasl \
        --without-winidn \
        --without-schannel \
        --without-secure-transport \
        --without-ca-bundle \
        --without-ca-path \
        --enable-optimize \
        --enable-warnings
    
    make -j$(nproc)
    make install
    
    # Create combined static library
    create_combined_static_lib_android $arch $install_prefix
    
    log_info "curl built successfully for Android $arch"
}

# Build curl for Apple platforms
build_curl_apple() {
    local platform=$1
    local arch=$2
    local install_prefix="$INSTALL_DIR/$platform/$arch"
    
    log_info "Building curl for $platform $arch..."
    
    cd "$WORK_DIR/curl-${CURL_VERSION}"
    make clean &> /dev/null || true
    
    get_apple_toolchain $platform $arch
    
    export PKG_CONFIG_PATH="$install_prefix/lib/pkgconfig"
    export CPPFLAGS="-I$install_prefix/include"
    export LDFLAGS="$LDFLAGS -L$install_prefix/lib"
    export LIBS="-lssl -lcrypto -lz"
    
    ./configure \
        --host="$arch-apple-darwin" \
        --prefix="$install_prefix" \
        --with-openssl="$install_prefix" \
        --with-zlib="$install_prefix" \
        --enable-static \
        --disable-shared \
        --disable-verbose \
        --disable-manual \
        --disable-ldap \
        --disable-ldaps \
        --disable-rtsp \
        --disable-dict \
        --disable-telnet \
        --disable-tftp \
        --disable-pop3 \
        --disable-imap \
        --disable-smb \
        --disable-smtp \
        --disable-gopher \
        --without-librtmp \
        --without-libidn2 \
        --without-libpsl \
        --without-nghttp2 \
        --without-brotli \
        --without-zstd \
        --without-ca-bundle \
        --without-ca-path \
        --enable-optimize
    
    make -j$(nproc)
    make install
    
    # Create combined static library
    create_combined_static_lib_apple $platform $arch $install_prefix
    
    log_info "curl built successfully for $platform $arch"
}

# Create combined static library for Android
create_combined_static_lib_android() {
    local arch=$1
    local install_prefix=$2
    
    log_info "Creating combined static library for Android $arch..."
    
    cd "$install_prefix/lib"
    
    mkdir -p temp_objects
    cd temp_objects
    
    # Extract all object files
    for lib in ../libcurl.a ../libssl.a ../libcrypto.a ../libz.a; do
        if [ -f "$lib" ]; then
            $AR x "$lib"
        fi
    done
    
    # Create combined library
    $AR rcs ../libcurl-combined.a *.o
    $RANLIB ../libcurl-combined.a
    
    cd ..
    rm -rf temp_objects
    
    log_info "Combined static library created for Android $arch"
}

# Create combined static library for Apple platforms
create_combined_static_lib_apple() {
    local platform=$1
    local arch=$2
    local install_prefix=$3
    
    log_info "Creating combined static library for $platform $arch..."
    
    cd "$install_prefix/lib"
    
    mkdir -p temp_objects
    cd temp_objects
    
    # Extract all object files
    for lib in ../libcurl.a ../libssl.a ../libcrypto.a ../libz.a; do
        if [ -f "$lib" ]; then
            ar x "$lib"
        fi
    done
    
    # Create combined library
    ar rcs ../libcurl-combined.a *.o
    ranlib ../libcurl-combined.a
    
    cd ..
    rm -rf temp_objects
    
    log_info "Combined static library created for $platform $arch"
}

# Create universal (fat) library for Apple platforms
create_universal_library() {
    local platform=$1
    
    log_info "Creating universal library for $platform..."
    
    local universal_dir="$INSTALL_DIR/$platform/universal"
    mkdir -p "$universal_dir/lib"
    mkdir -p "$universal_dir/include"
    
    # Copy headers (same for all architectures)
    local first_arch
    case $platform in
        ios|visionos)
            first_arch="arm64"
            ;;
    esac
    
    cp -r "$INSTALL_DIR/$platform/$first_arch/include/"* "$universal_dir/include/"
    
    # Create universal libraries
    local libs=("libcurl.a" "libssl.a" "libcrypto.a" "libz.a" "libcurl-combined.a")
    
    for lib in "${libs[@]}"; do
        local inputs=()
        for arch in "${IOS_ARCHS[@]}"; do
            if [ -f "$INSTALL_DIR/$platform/$arch/lib/$lib" ]; then
                inputs+=("$INSTALL_DIR/$platform/$arch/lib/$lib")
            fi
        done
        
        if [ ${#inputs[@]} -gt 0 ]; then
            lipo -create "${inputs[@]}" -output "$universal_dir/lib/$lib"
            log_info "Created universal $lib for $platform"
        fi
    done
}

# Build for specific platform
build_platform() {
    local platform=$1
    
    log_platform "Building for $platform"
    
    case $platform in
        android)
            for arch in "${ANDROID_ARCHS[@]}"; do
                log_info "Building for Android $arch"
                mkdir -p "$INSTALL_DIR/android/$arch"
                
                build_openssl_android $arch
                build_zlib_android $arch
                build_curl_android $arch
                
                log_info "Completed Android $arch"
            done
            ;;
        ios)
            for arch in "${IOS_ARCHS[@]}"; do
                log_info "Building for iOS $arch"
                mkdir -p "$INSTALL_DIR/ios/$arch"
                
                build_openssl_apple ios $arch
                build_zlib_apple ios $arch
                build_curl_apple ios $arch
                
                log_info "Completed iOS $arch"
            done
            create_universal_library ios
            ;;
        visionos)
            for arch in "${VISIONOS_ARCHS[@]}"; do
                log_info "Building for visionOS $arch"
                mkdir -p "$INSTALL_DIR/visionos/$arch"
                
                build_openssl_apple visionos $arch
                build_zlib_apple visionos $arch
                build_curl_apple visionos $arch
                
                log_info "Completed visionOS $arch"
            done
            create_universal_library visionos
            ;;
    esac
}

# Create summary
create_summary() {
    log_info "Creating build summary..."
    
    cat > "$INSTALL_DIR/BUILD_SUMMARY.txt" << EOF
Universal Curl Build Summary
============================
OpenSSL Version: $OPENSSL_VERSION
zlib Version: $ZLIB_VERSION
curl Version: $CURL_VERSION
Build Date: $(date)
Linking: STATIC (no external dependencies)

Built Platforms:
EOF
    
    for platform in "${PLATFORMS[@]}"; do
        echo "" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
        echo "$platform:" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
        
        case $platform in
            android)
                for arch in "${ANDROID_ARCHS[@]}"; do
                    echo "  $arch:" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                    if [ -f "$INSTALL_DIR/android/$arch/lib/libcurl-combined.a" ]; then
                        echo "    libcurl-combined.a: SUCCESS ($(du -h "$INSTALL_DIR/android/$arch/lib/libcurl-combined.a" | cut -f1))" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                    else
                        echo "    libcurl-combined.a: FAILED" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                    fi
                done
                ;;
            ios|visionos)
                for arch in "${IOS_ARCHS[@]}"; do
                    echo "  $arch:" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                    if [ -f "$INSTALL_DIR/$platform/$arch/lib/libcurl-combined.a" ]; then
                        echo "    libcurl-combined.a: SUCCESS ($(du -h "$INSTALL_DIR/$platform/$arch/lib/libcurl-combined.a" | cut -f1))" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                    else
                        echo "    libcurl-combined.a: FAILED" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                    fi
                done
                if [ -f "$INSTALL_DIR/$platform/universal/lib/libcurl-combined.a" ]; then
                    echo "  universal:" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                    echo "    libcurl-combined.a: SUCCESS ($(du -h "$INSTALL_DIR/$platform/universal/lib/libcurl-combined.a" | cut -f1))" >> "$INSTALL_DIR/BUILD_SUMMARY.txt"
                fi
                ;;
        esac
    done
    
    # Create usage instructions
    cat > "$INSTALL_DIR/USAGE.txt" << 'EOF'
Universal Curl Library Usage Guide
==================================

This build creates static curl libraries with embedded OpenSSL and zlib for:
- Android (arm64-v8a, armeabi-v7a, x86, x86_64)
- iOS (arm64 device, x86_64 simulator, universal fat library)
- visionOS (arm64 device, x86_64 simulator, universal fat library)

Directory Structure:
-------------------
├── android/
│   ├── arm64-v8a/
│   ├── armeabi-v7a/
│   ├── x86/
│   └── x86_64/
├── ios/
│   ├── arm64/
│   ├── x86_64/
│   └── universal/    # Fat binary for both architectures
└── visionos/
    ├── arm64/
    ├── x86_64/
    └── universal/    # Fat binary for both architectures

Android Integration:
-------------------
In your CMakeLists.txt:

add_library(curl STATIC IMPORTED)
set_target_properties(curl PROPERTIES IMPORTED_LOCATION
    ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libcurl-combined.a)
target_link_libraries(your_target curl)

iOS/visionOS Integration:
------------------------
For Xcode projects, add the universal library:
- Drag libcurl-combined.a to your project
- Add to "Link Binary With Libraries"
- Add include path to "Header Search Paths"

For specific architectures, use the individual arch libraries.

Swift Package Manager:
---------------------
Create a Package.swift with binaryTarget pointing to your libraries.

Headers:
--------
#include <curl/curl.h>

All libraries are completely static with no external dependencies.
EOF
}

# Main execution
main() {
    parse_arguments "$@"
    
    log_info "Universal curl build script started"
    log_info "Building for platforms: ${PLATFORMS[*]}"
    log_info "Work directory: $WORK_DIR"
    log_info "Install directory: $INSTALL_DIR"
    
    check_prerequisites
    download_sources
    extract_sources
    
    for platform in "${PLATFORMS[@]}"; do
        build_platform $platform
    done
    
    create_summary
    
    log_info "Build completed successfully!"
    log_info "Built libraries are available in: $INSTALL_DIR"
    log_info "See BUILD_SUMMARY.txt and USAGE.txt for details"
}

# Handle script interruption
cleanup() {
    log_info "Cleaning up..."
    # Uncomment to remove work directory
    # rm -rf "$WORK_DIR"
}

trap cleanup EXIT

# Run main function
main "$@"