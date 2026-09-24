#pragma once
#include "../gufo_posix.h"
#define WIFEXITED(s) (((s) & 0x7f) == 0)
#define WEXITSTATUS(s) (((s) >> 8) & 0xff)
#define WIFSIGNALED(s) (((s) & 0x7f) != 0)
#define WTERMSIG(s) ((s) & 0x7f)
#define WNOHANG 1
#ifdef __cplusplus
extern "C" {
#endif
int waitpid(int pid, int* status, int options);
#ifdef __cplusplus
}
#endif
