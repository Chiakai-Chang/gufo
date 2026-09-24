// Minimal POSIX layer for the Windows port. Only what Gufo calls is provided.
#pragma once
#if defined(_WIN32)

#include "gufo_prelude.h"

#include <errno.h>
#include <io.h>
#include <direct.h>
#include <process.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- fcntl -----------------------------------------------------------------
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
// O_DIRECT maps to FILE_FLAG_NO_BUFFERING. Only the wide-path open() overload
// and gufo_reopen_direct() honour it; the CRT never sees this bit.
#ifndef O_DIRECT
#define O_DIRECT 0x10000000
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0x40000000
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0
#endif

// ---- unistd ----------------------------------------------------------------
ssize_t pread(int fd, void* buf, size_t count, off_t offset);
ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset);
int fsync(int fd);
int ftruncate(int fd, off_t length);

#define _SC_PAGESIZE 30
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_NPROCESSORS_ONLN 84
#define _SC_NPROCESSORS_CONF 83
#define _SC_PHYS_PAGES 85
#define _SC_AVPHYS_PAGES 86
long sysconf(int name);

#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4
#define F_DUPFD 0
#define F_DUPFD_CLOEXEC 1030
#define FD_CLOEXEC 1
int fcntl(int fd, int cmd, ...);
// Opens with FILE_FLAG_NO_BUFFERING when flags has O_DIRECT, else like _wopen.
int gufo_open_wide(const wchar_t* path, int flags, int mode);
// Reopens an open file for unbuffered reads (Linux: open("/proc/self/fd/N", O_DIRECT)).
int gufo_reopen_direct(int fd);

// ---- mman ------------------------------------------------------------------
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_POPULATE 0x08000
#define MAP_NORESERVE 0x04000
#define MAP_FAILED ((void*)-1)
#define MADV_NORMAL 0
#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
#define MADV_POPULATE_READ 22
#define MADV_POPULATE_WRITE 23
void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void* addr, size_t length);
int madvise(void* addr, size_t length, int advice);
int mlock(const void* addr, size_t length);
int munlock(const void* addr, size_t length);

// ---- fadvise / flock -------------------------------------------------------
#define POSIX_FADV_NORMAL 0
#define POSIX_FADV_RANDOM 1
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
#define POSIX_FADV_NOREUSE 5
int posix_fadvise(int fd, off_t offset, off_t len, int advice);
#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8
int flock(int fd, int operation);
int mkstemp(char* tmpl);

// ---- resource / utsname ----------------------------------------------------
struct gufo_timeval_ru { long tv_sec; long tv_usec; };
struct rusage {
  struct gufo_timeval_ru ru_utime;
  struct gufo_timeval_ru ru_stime;
  long ru_maxrss;
  long ru_minflt;
  long ru_majflt;
  long ru_nvcsw;
  long ru_nivcsw;
};
#define RUSAGE_SELF 0
int getrusage(int who, struct rusage* usage);
struct utsname {
  char sysname[65];
  char nodename[65];
  char release[65];
  char version[65];
  char machine[65];
};
int uname(struct utsname* buf);

// ---- process / signals (media helpers only; unsupported on Windows) -------
#ifndef SIGPIPE
#define SIGPIPE 13
#endif
#ifndef SIGKILL
#define SIGKILL 9
#endif
#ifndef SIGTERM
#define SIGTERM 15
#endif
typedef int pid_t_gufo;
int kill(int pid, int sig);

#ifndef CLOCK_MONOTONIC
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
typedef int clockid_t;
int clock_gettime(clockid_t clk, struct timespec* ts);
#endif

// ---- *at family: unsupported, used only by the optional --cache-disk store --
#define AT_FDCWD -100
#define AT_REMOVEDIR 0x200
#define AT_SYMLINK_NOFOLLOW 0x100
#define UTIME_NOW ((1l << 30) - 1l)
#define UTIME_OMIT ((1l << 30) - 2l)
int openat(int dirfd, const char* path, int flags, ...);
int renameat(int olddirfd, const char* oldpath, int newdirfd, const char* newpath);
int unlinkat(int dirfd, const char* path, int flags);
struct _stat64;
int fstatat(int dirfd, const char* path, struct _stat64* st, int flags);
int utimensat(int dirfd, const char* path, const struct timespec times[2], int flags);
int geteuid(void);
int mincore(void* addr, size_t length, unsigned char* vec);
int dprintf(int fd, const char* format, ...);
struct _iobuf;
struct _iobuf* fmemopen(void* buf, size_t size, const char* mode);
int getuid(void);

#ifdef __cplusplus
}
#endif

#include <time.h>
static inline struct tm* gmtime_r(const time_t* t, struct tm* out) {
  return gmtime_s(out, t) == 0 ? out : 0;
}
static inline struct tm* localtime_r(const time_t* t, struct tm* out) {
  return localtime_s(out, t) == 0 ? out : 0;
}

// ---- stat: 64-bit sizes ------------------------------------------------------
#include <sys/types.h>
#include <sys/stat.h>
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISLNK
#define S_ISLNK(m) 0
#endif
#ifndef S_IRUSR
#define S_IRUSR _S_IREAD
#define S_IWUSR _S_IWRITE
#define S_IXUSR 0
#define S_IRWXU (_S_IREAD | _S_IWRITE)
#define S_IRWXG 0
#define S_IRWXO 0
#endif

#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif

#ifdef __cplusplus
// std::filesystem::path::c_str() is wchar_t on Windows.
inline int open(const wchar_t* path, int flags, int mode = _S_IREAD | _S_IWRITE) {
  return gufo_open_wide(path, flags, mode);
}
inline int _stat64(const wchar_t* path, struct _stat64* st) { return _wstat64(path, st); }
inline int chmod(const wchar_t* path, int mode) { return _wchmod(path, mode); }
inline int unlink(const wchar_t* path) { return _wunlink(path); }
inline int mkdir(const wchar_t* path, int) { return _wmkdir(path); }
inline int mkdir(const char* path, int) { return _mkdir(path); }
#endif

// struct stat / fstat / stat with 64-bit st_size.
#ifndef GUFO_NO_STAT64
#define stat _stat64
#define fstat _fstat64
#endif
#define lseek _lseeki64
#define lstat _stat64

#endif  // _WIN32
