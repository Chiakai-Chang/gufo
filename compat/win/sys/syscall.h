#pragma once
#include "../gufo_posix.h"
#define SYS_memfd_create 319
#define MFD_CLOEXEC 1
#ifdef __cplusplus
extern "C"
#endif
long syscall(long number, ...);
