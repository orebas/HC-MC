Package: cppad:x64-linux@20240000.7

**Host Environment**

- Host: x64-linux
- Compiler: Clang 18.1.3
-    vcpkg-tool version: 2024-11-12-eb492805e92a2c14a230f5c3deb3e89f6771c321
    vcpkg-scripts version: 9b5cb8e55 2024-11-14 (8 days ago)

**To Reproduce**

`vcpkg install `

**Failure logs**

```
Downloading coin-or-CppAD-20240000.7.tar.gz
Successfully downloaded coin-or-CppAD-20240000.7.tar.gz.
/source/src/vcpkg/metrics.cpp(169): unreachable code was reached
CMake Error at scripts/cmake/vcpkg_download_distfile.cmake:231 (message):
  Download failed, halting portfile.
Call Stack (most recent call first):
  scripts/cmake/vcpkg_from_github.cmake:106 (vcpkg_download_distfile)
  buildtrees/versioning_/versions/cppad/d69e902bac2437d6ccb8828d183d1d00d5ff8a4b/portfile.cmake:1 (vcpkg_from_github)
  scripts/ports.cmake:192 (include)



```

**Additional context**

<details><summary>vcpkg.json</summary>

```
{
  "name": "hc-mc",
  "version": "1.0.0",
  "description": "A testbed for homotopy continuation methods",
  "dependencies": [
    "eigen3",
    "cppad",
    {
      "name": "doctest",
      "platform": "!windows"
    }
  ],
  "builtin-baseline": "d221c5d2cbadf35ceb266cbb95750a940b103b65"
}

```
</details>
