#pragma once
#include_next <signal.h>
#include "gufo_posix.h"
typedef int sigset_t;
#ifndef SIGBUS
#define SIGBUS 10  // no bus-error signal on Windows; sigaction() ignores it
#endif
#define SA_RESETHAND 0x80000000
struct sigaction {
  void (*sa_handler)(int);
  sigset_t sa_mask;
  int sa_flags;
};
#ifdef __cplusplus
extern "C" {
#endif
int sigaction(int sig, const struct sigaction* act, struct sigaction* old);
static inline int sigemptyset(sigset_t* set) { *set = 0; return 0; }
static inline int sigaddset(sigset_t* set, int sig) { *set |= (1 << sig); return 0; }
#ifdef __cplusplus
}
#endif
