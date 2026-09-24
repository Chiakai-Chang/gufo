// Windows port prelude, force-included before every C/C++/HIP translation unit.
// It fixes the POSIX types and macros that must exist before the MSVC CRT headers.
#pragma once
#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_NONSTDC_NO_WARNINGS
#define _CRT_NONSTDC_NO_WARNINGS
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// 64-bit off_t: the MSVC default is a 32-bit long, which breaks files over 2 GiB.
#ifndef _OFF_T_DEFINED
#define _OFF_T_DEFINED
typedef long _off_t;
typedef long long off_t;
#endif

#include <stddef.h>
#include <stdint.h>
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef long long ssize_t;
#endif

#ifndef _PID_T_DEFINED_GUFO
#define _PID_T_DEFINED_GUFO
typedef int pid_t;
#endif
typedef int uid_t;
typedef int gid_t;
typedef unsigned int mode_t;

#endif  // _WIN32
