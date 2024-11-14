#!/usr/bin/env bash
set -eo pipefail

# Print usage information
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --debug                Build with debug symbols (default)"
    echo "  --release              Build with optimizations"
    echo "  --compiler=<name>      Use specific compiler (gcc|clang)"
    echo "  --compiler-version=<v> Use specific compiler version"
    echo "  --clean               Clean build directory before building"
    echo "  --help                Show this help message"
}

# Default values
BUILD_TYPE="Debug"
COMPILER="gcc"
COMPILER_VERSION=""
CLEAN=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --compiler=*)
            COMPILER="${1#*=}"
            shift
            ;;
        --compiler-version=*)
            COMPILER_VERSION="${1#*=}"
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Install system dependencies
install_system_dependencies() {
    echo "Checking and installing system dependencies..."
    
    # Check if we have sudo access
    if ! command -v sudo &> /dev/null; then
        echo "Error: sudo is not available. Please run as root or install sudo."
        exit 1
    fi

    # Update package list
    sudo apt-get update

    # Install required packages
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        libgmp-dev \
        libmpfr-dev \
        libcppad-dev \
        libeigen3-dev \
        libboost-all-dev \
        libtool

    if [ "$COMPILER" = "clang" ]; then
        if [ -n "$COMPILER_VERSION" ]; then
            sudo apt-get install -y clang-$COMPILER_VERSION
        else
            sudo apt-get install -y clang
        fi
    fi
}

# Setup compiler environment
setup_compiler() {
    echo "Setting up compiler environment..."
    
    case $COMPILER in
        gcc)
            if [ -n "$COMPILER_VERSION" ]; then
                export CC="gcc-${COMPILER_VERSION}"
                export CXX="g++-${COMPILER_VERSION}"
            else
                export CC="gcc"
                export CXX="g++"
            fi
            ;;
            
        clang)
            if [ -n "$COMPILER_VERSION" ]; then
                export CC="clang-${COMPILER_VERSION}"
                export CXX="clang++-${COMPILER_VERSION}"
            else
                export CC="clang"
                export CXX="clang++"
            fi
            ;;
            
        *)
            echo "Unsupported compiler: $COMPILER"
            exit 1
            ;;
    esac
    
    # Verify compiler exists
    if ! command -v "$CC" >/dev/null 2>&1; then
        echo "Error: Compiler $CC not found"
        exit 1
    fi
    
    if ! command -v "$CXX" >/dev/null 2>&1; then
        echo "Error: Compiler $CXX not found"
        exit 1
    fi
    
    echo "Using C compiler: $CC ($(command -v "$CC"))"
    echo "Using C++ compiler: $CXX ($(command -v "$CXX"))"
}

# Setup vcpkg
setup_vcpkg() {
    echo "Setting up vcpkg..."
    
    # Only clone if directory doesn't exist
    if [ ! -d "external/vcpkg" ]; then
        echo "Cloning vcpkg..."
        mkdir -p external
        if ! git clone https://github.com/Microsoft/vcpkg.git external/vcpkg; then
            echo "Failed to clone vcpkg repository"
            exit 1
        fi
    fi

    # Verify the bootstrap script exists
    if [ ! -f "external/vcpkg/bootstrap-vcpkg.sh" ]; then
        echo "bootstrap-vcpkg.sh not found. vcpkg installation may be corrupted."
        echo "Delete the external/vcpkg directory and try again."
        exit 1
    fi

    # Make bootstrap script executable
    chmod +x external/vcpkg/bootstrap-vcpkg.sh

    # Only bootstrap if vcpkg executable doesn't exist
    if [ ! -f "external/vcpkg/vcpkg" ]; then
        echo "Bootstrapping vcpkg..."
        if ! ./external/vcpkg/bootstrap-vcpkg.sh; then
            echo "Failed to bootstrap vcpkg"
            exit 1
        fi
    fi

    # Set VCPKG_ROOT
    export VCPKG_ROOT="$(pwd)/external/vcpkg"

    # Install dependencies from manifest
    echo "Installing dependencies from vcpkg.json manifest..."
    if ! "$VCPKG_ROOT/vcpkg" install --triplet x64-linux; then
        echo "Failed to install dependencies from manifest"
        exit 1
    fi
}

# Main build process
main() {
    # Install system dependencies
    install_system_dependencies

    # Setup compiler
    setup_compiler

    # Setup vcpkg
    setup_vcpkg

    # Clean build directory if requested
    if [ "$CLEAN" -eq 1 ]; then
        echo "Cleaning build directory..."
        rm -rf build
    fi

    # Create build directory
    mkdir -p build

    echo "Configuring with CMake..."
    cmake -B build -S . \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET=x64-linux \
        -DBUILD_TESTING=ON \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX"

    echo "Building..."
    cmake --build build -j$(nproc)

    echo "Running tests..."
    ctest --test-dir build --output-on-failure

    echo "Build complete!"
    echo "Type: ${BUILD_TYPE}"
    if [ "$BUILD_TYPE" = "Debug" ]; then
        echo "Debug symbols are enabled. You can use gdb/lldb for debugging."
    fi
}

# Run main function
main