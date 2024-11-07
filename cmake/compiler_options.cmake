# MIT License 
# Copyright (c) 2018-Today Michele Adduci <adduci@tutanota.com>
#
# Compiler options with hardening flags

if(MSVC)
    list(APPEND compiler_options 
        /W4
        /WX
        /permissive-
        $<$<CONFIG:RELEASE>:/O2 /Ob2 >
        $<$<CONFIG:MINSIZEREL>:/O1 /Ob1>
        $<$<CONFIG:RELWITHDEBINFO>:/Zi /O2 /Ob1>
        $<$<CONFIG:DEBUG>:/Zi /Ob0 /Od /RTC1>)

    list(APPEND compiler_definitions
        _UNICODE
        WINDOWS
        $<$<OR:$<CONFIG:RELEASE>,$<CONFIG:RELWITHDEBINFO>,$<CONFIG:MINSIZEREL>>:NDEBUG>
        $<$<CONFIG:DEBUG>:_DEBUG>)

    list(APPEND linker_flags
        $<$<BOOL:${BUILD_SHARED_LIBS}>:/LTCG>
    )

    set(MSVC_RUNTIME_TYPE $<IF:$<BOOL:${BUILD_WITH_MT}>,MultiThreaded$<$<CONFIG:Debug>:Debug>,MultiThreaded$<$<CONFIG:Debug>:Debug>>DLL)

else(MSVC)
    # Enhanced warning flags for GCC/Clang
    list(APPEND compiler_options 
        -Wall
        -Wextra
        -Wpedantic
        -Werror=return-type
        -Werror=uninitialized
        -Werror=maybe-uninitialized
        -Wconversion
        -Wsign-conversion
        -Wcast-align
        -Wcast-qual
        -Wdisabled-optimization
        -Wformat=2
        -Winit-self
        -Wlogical-op
        -Wmissing-include-dirs
        -Wnoexcept
        -Wold-style-cast
        -Woverloaded-virtual
        -Wredundant-decls
        -Wshadow
        -Wsign-promo
        -Wstrict-null-sentinel
        -Wstrict-overflow=5
        -Wswitch-default
        -Wundef
        $<$<CXX_COMPILER_ID:GNU>:-Wuseless-cast>
        -Wno-unknown-pragmas
        $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>
        $<$<CONFIG:DEBUG>:-O0 -g3>
        $<$<CONFIG:RELEASE>:-O3>
    )

    # Enhanced security flags
    list(APPEND compiler_definitions
        $<$<OR:$<CONFIG:RELEASE>,$<CONFIG:MINSIZEREL>>:_FORTIFY_SOURCE=2>
    )
 
    list(APPEND linker_flags
        $<$<NOT:$<CXX_COMPILER_ID:AppleClang>>:-Wl,-z,defs>
        $<$<NOT:$<CXX_COMPILER_ID:AppleClang>>:-Wl,-z,now>
        $<$<NOT:$<CXX_COMPILER_ID:AppleClang>>:-Wl,-z,relro>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:AppleClang>>,$<NOT:$<CXX_COMPILER_ID:Clang>>,$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>>:-Wl,-pie>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:AppleClang>>,$<NOT:$<CXX_COMPILER_ID:Clang>>,$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>>:-fpie>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:AppleClang>>,$<NOT:$<CXX_COMPILER_ID:Clang>>,$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>>:-pipe>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:AppleClang>>,$<NOT:$<CXX_COMPILER_ID:Clang>>,$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>>:-static-libstdc++>
        $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>
        $<$<CONFIG:DEBUG>:-fsanitize=address>
        $<$<CONFIG:DEBUG>:-fsanitize=leak>
        $<$<CONFIG:DEBUG>:-fsanitize=undefined>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:AppleClang>>,$<NOT:$<CXX_COMPILER_ID:Clang>>>:-fstack-clash-protection>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:AppleClang>>,$<NOT:$<CXX_COMPILER_ID:Clang>>>:-fbounds-check>
        -fstack-protector
        -fPIC
    )

endif()

# Make sure the options are actually applied to targets
function(target_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE ${compiler_options})
    else()
        target_compile_options(${target_name} PRIVATE ${compiler_options})
    endif()
endfunction()