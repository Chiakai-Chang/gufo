// posix_spawn is not available on Windows. Media helpers that launch FFmpeg fail at runtime.
#pragma once
#include "gufo_posix.h"

typedef struct { int unused; } posix_spawn_file_actions_t;
typedef struct { int unused; } posix_spawnattr_t;
#ifdef __cplusplus
extern "C" {
#endif
int posix_spawnp(pid_t* pid, const char* file, const posix_spawn_file_actions_t* fa, const posix_spawnattr_t* attr, char* const argv[], char* const envp[]);
int posix_spawn(pid_t* pid, const char* path, const posix_spawn_file_actions_t* fa, const posix_spawnattr_t* attr, char* const argv[], char* const envp[]);
int posix_spawn_file_actions_init(posix_spawn_file_actions_t* fa);
int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t* fa);
int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t* fa, int fd, int newfd);
int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t* fa, int fd);
int posix_spawn_file_actions_addopen(posix_spawn_file_actions_t* fa, int fd, const char* path, int oflag, int mode);
int pipe(int fds[2]);
#ifdef __cplusplus
}
#endif
