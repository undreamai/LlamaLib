#!/bin/bash

# This script builds OpenSSL for Android

set -e

# Configuration
ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-$HOME/Android/Sdk/ndk/25.2.9519653}
ANDROID_API_LEVEL=${ANDROID_API_LEVEL:-21}
WORK_DIR=${WORK_DIR:-/tmp/android-openssl-build}
INSTALL_DIR=${INSTALL_DIR:-$(pwd)/openssl}

# Version configurations
OPENSSL_VERSION="3.3.3"
OPENSSL_IOS_VERSION="3.3.3001"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


# Download source files
download_openssl_source() {
    log_info "Downloading OpenSSL ${OPENSSL_VERSION}..."
    cd $WORK_DIR

    link="https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz"
    archive=`basename $link`
    if [ ! -f $archive ]; then wget $link; fi

    log_info "Source files downloaded"
    log_info "Extracting source files..."
    
    tar -xzf "$archive"
    
    log_info "Source files extracted"
}

# Get Android toolchain info for architecture
get_android_toolchain() {
    local arch=$1
    local api_level=$2
    
    case $arch in
        android_arm64)
            export ANDROID_ARCH="aarch64"
            export ANDROID_EABI="aarch64-linux-android"
            export OPENSSL_ARCH="android-arm64"
            ;;
        android_x64)
            export ANDROID_ARCH="x86_64"
            export ANDROID_EABI="x86_64-linux-android"
            export OPENSSL_ARCH="android-x86_64"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    export TOOLCHAIN_ROOT="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64"
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

# Build OpenSSL for specific architecture
build_openssl_android() {
    download_openssl_source

    local arch=$1
    local install_prefix="$INSTALL_DIR/$arch"
    
    log_info "Building OpenSSL for $arch..."
    
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
        -static &> $arch.log
    
    make -j$(nproc) &> $arch.log
    make install_sw &> $arch.log

    rm -r $install_prefix/bin
    
    log_info "OpenSSL built successfully for $arch"
}

# Download prebuilt OpenSSL for iOS/visionOS
download_openssl_ios() {
    local arch=$1
    local install_prefix="$INSTALL_DIR/$arch"

    log_info "Downloading prebuild OpenSSL ${OPENSSL_VERSION}..."

    link="https://github.com/krzyzanowskim/OpenSSL/archive/refs/tags/$OPENSSL_IOS_VERSION.zip"
    archive=`basename $link`
    if [ ! -f $archive ]; then wget $link; fi

    log_info "Source files downloaded"
    log_info "Extracting source files..."
    
    unzip -n $archive
    
    log_info "Source files extracted"
    case $arch in
        ios)
            copyarch=iphoneos
            ;;
        visionos)
            copyarch=visionos
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    mkdir -p $install_prefix/include/openssl $install_prefix/lib
    cp OpenSSL*/$copyarch/include/OpenSSL/* $install_prefix/include/openssl/
    cp OpenSSL*/$copyarch/lib/* $install_prefix/lib/

}

# Main execution
main() {
    arch=$1
    log_info "Building for architecture: $arch" 
    mkdir -p $WORK_DIR $INSTALL_DIR/$arch

    case $arch in
        android_arm64)
            build_openssl_android $arch
            ;;
        android_x64)
            build_openssl_android $arch
            ;;
        ios)
            download_openssl_ios $arch
            ;;
        visionos)
            download_openssl_ios $arch
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
}

main "$@"