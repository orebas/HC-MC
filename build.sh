#!/bin/bash
set -e  # Exit on any error

# Parse command line arguments
BUILD_TYPE="Debug"  # Default to Debug
COMPILER="gcc"      # Default to gcc
COMPILER_VERSION="" # Optional version specifier

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --debug                Build with debug symbols (default)"
    echo "  --release              Build with optimizations"
    echo "  --compiler=<name>      Use specific compiler (gcc|clang)"
    echo "  --compiler-version=<v> Use specific compiler version (e.g., 13, 15)"
    echo "  --help                 Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --compiler=clang --compiler-version=15 --release"
    echo "  $0 --compiler=gcc --compiler-version=13"
}

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

# Set up compiler environment variables
setup_compiler() {
    local compiler_name="$1"
    local version="$2"
    
    # Clear any existing compiler variables
    unset CC CXX
    
    case $compiler_name in
        gcc)
            if [ -n "$version" ]; then
                export CC="gcc-${version}"
                export CXX="g++-${version}"
            else
                export CC="gcc"
                export CXX="g++"
            fi
            ;;
            
        clang)
            if [ -n "$version" ]; then
                export CC="clang-${version}"
                export CXX="clang++-${version}"
            else
                export CC="clang"
                export CXX="clang++"
            fi
            ;;
            
        *)
            echo "Unsupported compiler: $compiler_name"
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

# Set up the compiler
setup_compiler "$COMPILER" "$COMPILER_VERSION"

# Clone vcpkg if it doesn't exist
if [ ! -d "external/vcpkg" ]; then
    echo "Cloning vcpkg..."
    git clone https://github.com/Microsoft/vcpkg.git external/vcpkg
    external/vcpkg/bootstrap-vcpkg.sh
fi

# Set VCPKG_ROOT to the submodule path
export VCPKG_ROOT=$(pwd)/external/vcpkg

# Clean build directory
rm -rf build

echo "Configuring for ${BUILD_TYPE} build using ${CXX}..."

# Configure the project with CMake and vcpkg
cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_TARGET_TRIPLET=x64-linux \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX"

# Build the project
echo "Building..."
cmake --build build

# Optional: Run tests after building
echo "Running tests..."
ctest --test-dir build --output-on-failure

echo "Build complete. Type: ${BUILD_TYPE}"
if [ "$BUILD_TYPE" = "Debug" ]; then
    echo "Debug symbols are enabled. You can use gdb/lldb for debugging."
fi

# Print compiler version information
$CXX --version