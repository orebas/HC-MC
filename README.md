## Building with vcpkg

This project uses vcpkg for dependency management. To build:

1. Install vcpkg if you haven't already:
```bash
git clone https://github.com/Microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh  # or bootstrap-vcpkg.bat on Windows
```

2. Build the project:
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

Replace `[path to vcpkg]` with the actual path where you installed vcpkg.

For convenience, you can set the `VCPKG_ROOT` environment variable to your vcpkg installation directory:
```bash
export VCPKG_ROOT=/path/to/vcpkg
```

Then build with:
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build
```








