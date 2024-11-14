Package: python3[core,extensions]:x64-linux@3.11.10

**Host Environment**

- Host: x64-linux
- Compiler: GNU 13.2.0
-    vcpkg-tool version: 2024-11-12-eb492805e92a2c14a230f5c3deb3e89f6771c321
    vcpkg-scripts version: 772f784ba 2024-11-13 (5 hours ago)

**To Reproduce**

`vcpkg install `

**Failure logs**

```
CMake Warning at buildtrees/versioning_/versions/python3/62c1ff180dae6af1ba4aff966bd87f9683c6d8f1/portfile.cmake:16 (message):
  python3 currently requires the following programs from the system package
  manager:

      autoconf automake autoconf-archive

  On Debian and Ubuntu derivatives:

      sudo apt-get install autoconf automake autoconf-archive

  On recent Red Hat and Fedora derivatives:

      sudo dnf install autoconf automake autoconf-archive

  On Arch Linux and derivatives:

      sudo pacman -S autoconf automake autoconf-archive

  On Alpine:

      apk add autoconf automake autoconf-archive

  On macOS:

      brew install autoconf automake autoconf-archive

Call Stack (most recent call first):
  scripts/ports.cmake:192 (include)


Downloading python-cpython-v3.11.10.tar.gz
Successfully downloaded python-cpython-v3.11.10.tar.gz.
-- Extracting source /home/orebas/cpp/HC-MC/external/vcpkg/downloads/python-cpython-v3.11.10.tar.gz
-- Applying patch 0001-only-build-required-projects.patch
-- Applying patch 0003-use-vcpkg-zlib.patch
-- Applying patch 0004-devendor-external-dependencies.patch
-- Applying patch 0005-dont-copy-vcruntime.patch
-- Applying patch 0008-python.pc.patch
-- Applying patch 0010-dont-skip-rpath.patch
-- Applying patch 0012-force-disable-modules.patch
-- Applying patch 0014-fix-get-python-inc-output.patch
-- Applying patch 0015-dont-use-WINDOWS-def.patch
-- Applying patch 0016-undup-ffi-symbols.patch
-- Applying patch 0018-fix-sysconfig-include.patch
-- Applying patch 0019-fix-ssl-linkage.patch
-- Applying patch 0002-static-library.patch
-- Applying patch 0011-gcc-ldflags-fix.patch
-- Using source at /home/orebas/cpp/HC-MC/external/vcpkg/buildtrees/python3/src/v3.11.10-0e790061b6.clean
-- Getting CMake variables for x64-linux-dbg
-- Getting CMake variables for x64-linux-rel
-- Generating configure for x64-linux
CMake Error at scripts/cmake/vcpkg_execute_required_process.cmake:127 (message):
    Command failed: /usr/bin/autoreconf -vfi
    Working Directory: /home/orebas/cpp/HC-MC/external/vcpkg/buildtrees/python3/src/v3.11.10-0e790061b6.clean/
    Error code: 1
    See logs for more information:
      /home/orebas/cpp/HC-MC/external/vcpkg/buildtrees/python3/autoconf-x64-linux-err.log

Call Stack (most recent call first):
  scripts/cmake/vcpkg_configure_make.cmake:731 (vcpkg_execute_required_process)
  buildtrees/versioning_/versions/python3/62c1ff180dae6af1ba4aff966bd87f9683c6d8f1/portfile.cmake:293 (vcpkg_configure_make)
  scripts/ports.cmake:192 (include)



```

<details><summary>/home/orebas/cpp/HC-MC/external/vcpkg/buildtrees/python3/autoconf-x64-linux-err.log</summary>

```
autoreconf: export WARNINGS=
autoreconf: Entering directory '.'
autoreconf: configure.ac: not using Gettext
autoreconf: running: aclocal --force 
autoreconf: configure.ac: tracing
autoreconf: configure.ac: not using Libtool
autoreconf: configure.ac: not using Intltool
autoreconf: configure.ac: not using Gtkdoc
autoreconf: running: /usr/bin/autoconf --force
configure.ac:18: error: possibly undefined macro: AC_MSG_ERROR
      If this token and others are legitimate, please use m4_pattern_allow.
      See the Autoconf documentation.
autoreconf: error: /usr/bin/autoconf failed with exit status: 1
```
</details>

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
    },
    "boost"
  ],
  "builtin-baseline": "d221c5d2cbadf35ceb266cbb95750a940b103b65"
}

```
</details>
