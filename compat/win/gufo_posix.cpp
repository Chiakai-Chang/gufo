// Win32 implementation of the POSIX subset declared in gufo_posix.h / gufo_socket.h.
#if defined(_WIN32)

#include "gufo_posix.h"
#include "gufo_socket.h"
#include "sys/wait.h"
#include "spawn.h"
#include <signal.h>

#include <windows.h>
#include <psapi.h>

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <mutex>

namespace {

void IgnoreInvalidParameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned int, uintptr_t) {}

// Open every CRT descriptor in binary mode (POSIX has no text mode), and let
// CRT calls on a bad descriptor fail with EBADF instead of aborting.
struct CrtInit {
  CrtInit() {
    _set_fmode(_O_BINARY);
    _set_invalid_parameter_handler(IgnoreInvalidParameter);
  }
} g_crt_init;

HANDLE FdHandle(int fd) {
  return reinterpret_cast<HANDLE>(_get_osfhandle(fd));
}

void SetErrnoFromWin32(DWORD error) {
  switch (error) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND: errno = ENOENT; break;
    case ERROR_ACCESS_DENIED: errno = EACCES; break;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
    case ERROR_COMMITMENT_LIMIT: errno = ENOMEM; break;
    case ERROR_INVALID_HANDLE: errno = EBADF; break;
    case ERROR_HANDLE_EOF: errno = 0; break;
    default: errno = EINVAL; break;
  }
}

struct MappingInfo {
  void* base;  // value MapViewOfFile / VirtualAlloc returned
  bool virtual_alloc;
};
std::mutex g_map_mutex;
std::map<void*, MappingInfo> g_maps;  // user pointer -> mapping

DWORD AllocationGranularity() {
  static const DWORD value = [] {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwAllocationGranularity;
  }();
  return value;
}

}  // namespace

extern "C" {

ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
  HANDLE h = FdHandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  size_t done = 0;
  auto* out = static_cast<char*>(buf);
  while (done < count) {
    const DWORD want = static_cast<DWORD>(
        (count - done) > 0x40000000u ? 0x40000000u : (count - done));
    OVERLAPPED ov{};
    const unsigned long long pos = static_cast<unsigned long long>(offset) + done;
    ov.Offset = static_cast<DWORD>(pos & 0xffffffffu);
    ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
    DWORD got = 0;
    if (!ReadFile(h, out + done, want, &got, &ov)) {
      const DWORD err = GetLastError();
      if (err == ERROR_HANDLE_EOF)
        break;
      SetErrnoFromWin32(err);
      return done > 0 ? static_cast<ssize_t>(done) : -1;
    }
    if (got == 0)
      break;
    done += got;
    if (got < want)
      break;
  }
  return static_cast<ssize_t>(done);
}

ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset) {
  HANDLE h = FdHandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  size_t done = 0;
  const auto* in = static_cast<const char*>(buf);
  while (done < count) {
    const DWORD want = static_cast<DWORD>(
        (count - done) > 0x40000000u ? 0x40000000u : (count - done));
    OVERLAPPED ov{};
    const unsigned long long pos = static_cast<unsigned long long>(offset) + done;
    ov.Offset = static_cast<DWORD>(pos & 0xffffffffu);
    ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
    DWORD put = 0;
    if (!WriteFile(h, in + done, want, &put, &ov)) {
      SetErrnoFromWin32(GetLastError());
      return done > 0 ? static_cast<ssize_t>(done) : -1;
    }
    done += put;
    if (put < want)
      break;
  }
  return static_cast<ssize_t>(done);
}

int fcntl(int fd, int cmd, ...) {
  switch (cmd) {
    case F_DUPFD:
    case F_DUPFD_CLOEXEC:
      return _dup(fd);
    case F_GETFD:
    case F_GETFL:
      return FdHandle(fd) == INVALID_HANDLE_VALUE ? (errno = EBADF, -1) : 0;
    case F_SETFD:
    case F_SETFL:
      return 0;
    default:
      errno = EINVAL;
      return -1;
  }
}

namespace {
int OpenUnbuffered(const wchar_t* path) {
  HANDLE h = CreateFileW(path, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    SetErrnoFromWin32(GetLastError());
    return -1;
  }
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(h), _O_RDONLY | _O_BINARY);
  if (fd < 0)
    CloseHandle(h);
  return fd;
}
}  // namespace

int gufo_open_wide(const wchar_t* path, int flags, int mode) {
  if (flags & O_DIRECT) {
    if ((flags & (_O_WRONLY | _O_RDWR | _O_CREAT)) == 0)
      return OpenUnbuffered(path);
    flags &= ~O_DIRECT;
  }
  if (flags & O_DIRECTORY) {
    errno = ENOTSUP;
    return -1;
  }
  return _wopen(path, flags | _O_BINARY, mode);
}

int gufo_reopen_direct(int fd) {
  HANDLE h = FdHandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  wchar_t path[32768];
  const DWORD n = GetFinalPathNameByHandleW(h, path, 32768, FILE_NAME_NORMALIZED);
  if (n == 0 || n >= 32768) {
    SetErrnoFromWin32(GetLastError());
    return -1;
  }
  return OpenUnbuffered(path);
}

int fsync(int fd) {
  HANDLE h = FdHandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  if (!FlushFileBuffers(h)) {
    // Read-only handles cannot be flushed; POSIX fsync succeeds there.
    if (GetLastError() == ERROR_ACCESS_DENIED)
      return 0;
    SetErrnoFromWin32(GetLastError());
    return -1;
  }
  return 0;
}

int ftruncate(int fd, off_t length) {
  const errno_t err = _chsize_s(fd, length);
  if (err != 0) {
    errno = err;
    return -1;
  }
  return 0;
}

long sysconf(int name) {
  switch (name) {
    case _SC_PAGESIZE: {
      SYSTEM_INFO info;
      GetSystemInfo(&info);
      return static_cast<long>(info.dwPageSize);
    }
    case _SC_NPROCESSORS_ONLN:
    case _SC_NPROCESSORS_CONF:
      return static_cast<long>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    case _SC_PHYS_PAGES:
    case _SC_AVPHYS_PAGES: {
      MEMORYSTATUSEX status{};
      status.dwLength = sizeof(status);
      if (!GlobalMemoryStatusEx(&status))
        return -1;
      SYSTEM_INFO info;
      GetSystemInfo(&info);
      const unsigned long long bytes =
          name == _SC_PHYS_PAGES ? status.ullTotalPhys : status.ullAvailPhys;
      return static_cast<long>(bytes / info.dwPageSize);
    }
    default:
      errno = EINVAL;
      return -1;
  }
}

void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
  (void)addr;
  if (length == 0) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  if (flags & MAP_ANONYMOUS) {
    const DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
    void* p = VirtualAlloc(nullptr, length, MEM_RESERVE | MEM_COMMIT, protect);
    if (p == nullptr) {
      SetErrnoFromWin32(GetLastError());
      return MAP_FAILED;
    }
    std::lock_guard<std::mutex> lock(g_map_mutex);
    g_maps[p] = MappingInfo{p, true};
    return p;
  }

  HANDLE file = FdHandle(fd);
  if (file == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return MAP_FAILED;
  }
  const bool write = (prot & PROT_WRITE) != 0;
  const bool shared = (flags & MAP_SHARED) != 0;
  DWORD page_protect = PAGE_READONLY;
  DWORD view_access = FILE_MAP_READ;
  if (write && shared) {
    page_protect = PAGE_READWRITE;
    view_access = FILE_MAP_WRITE;
  } else if (write) {
    page_protect = PAGE_WRITECOPY;
    view_access = FILE_MAP_COPY;
  }
  HANDLE mapping = CreateFileMappingW(file, nullptr, page_protect, 0, 0, nullptr);
  if (mapping == nullptr) {
    SetErrnoFromWin32(GetLastError());
    return MAP_FAILED;
  }
  // Views must start on the allocation granularity (64 KiB); POSIX only needs a page.
  const unsigned long long gran = AllocationGranularity();
  const unsigned long long aligned = (static_cast<unsigned long long>(offset) / gran) * gran;
  const size_t skip = static_cast<size_t>(static_cast<unsigned long long>(offset) - aligned);
  void* base = MapViewOfFile(mapping, view_access, static_cast<DWORD>(aligned >> 32),
                             static_cast<DWORD>(aligned & 0xffffffffu), length + skip);
  const DWORD map_error = GetLastError();
  CloseHandle(mapping);  // the view keeps the mapping object alive
  if (base == nullptr) {
    SetErrnoFromWin32(map_error);
    return MAP_FAILED;
  }
  void* user = static_cast<char*>(base) + skip;
  std::lock_guard<std::mutex> lock(g_map_mutex);
  g_maps[user] = MappingInfo{base, false};
  return user;
}

int munmap(void* addr, size_t length) {
  (void)length;
  MappingInfo info{};
  {
    std::lock_guard<std::mutex> lock(g_map_mutex);
    auto it = g_maps.find(addr);
    if (it == g_maps.end()) {
      errno = EINVAL;
      return -1;
    }
    info = it->second;
    g_maps.erase(it);
  }
  const BOOL ok = info.virtual_alloc ? VirtualFree(info.base, 0, MEM_RELEASE)
                                     : UnmapViewOfFile(info.base);
  if (!ok) {
    SetErrnoFromWin32(GetLastError());
    return -1;
  }
  return 0;
}

int madvise(void* addr, size_t length, int advice) {
  if (advice == MADV_POPULATE_READ || advice == MADV_WILLNEED) {
    WIN32_MEMORY_RANGE_ENTRY range{addr, length};
    (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    if (advice == MADV_POPULATE_READ) {
      // PrefetchVirtualMemory is asynchronous; touch one byte per page so the range is resident on return.
      SYSTEM_INFO info;
      GetSystemInfo(&info);
      const size_t page = info.dwPageSize;
      volatile const char* p = static_cast<const char*>(addr);
      char sink = 0;
      for (size_t off = 0; off < length; off += page)
        sink ^= p[off];
      (void)sink;
    }
  } else if (advice == MADV_DONTNEED) {
    // Only drop pages from the working set; the mapping and its data stay valid.
    (void)VirtualUnlock(addr, length);
  }
  return 0;
}

int mlock(const void* addr, size_t length) {
  return VirtualLock(const_cast<void*>(addr), length) ? 0 : -1;
}

int munlock(const void* addr, size_t length) {
  return VirtualUnlock(const_cast<void*>(addr), length) ? 0 : -1;
}

int posix_fadvise(int fd, off_t offset, off_t len, int advice) {
  (void)fd; (void)offset; (void)len; (void)advice;
  return 0;
}

int flock(int fd, int operation) {
  HANDLE h = FdHandle(fd);
  if (h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return -1;
  }
  OVERLAPPED ov{};
  if (operation & LOCK_UN) {
    if (!UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov)) {
      SetErrnoFromWin32(GetLastError());
      return -1;
    }
    return 0;
  }
  DWORD lock_flags = 0;
  if (operation & LOCK_EX)
    lock_flags |= LOCKFILE_EXCLUSIVE_LOCK;
  if (operation & LOCK_NB)
    lock_flags |= LOCKFILE_FAIL_IMMEDIATELY;
  if (!LockFileEx(h, lock_flags, 0, MAXDWORD, MAXDWORD, &ov)) {
    const DWORD err = GetLastError();
    errno = err == ERROR_LOCK_VIOLATION ? EWOULDBLOCK : EINVAL;
    return -1;
  }
  return 0;
}

int mkstemp(char* tmpl) {
  const size_t n = strlen(tmpl);
  for (int attempt = 0; attempt < 100; ++attempt) {
    if (_mktemp_s(tmpl, n + 1) != 0)
      return -1;
    const int fd = _open(tmpl, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY | _O_NOINHERIT,
                         _S_IREAD | _S_IWRITE);
    if (fd >= 0)
      return fd;
    if (errno != EEXIST)
      return -1;
    memcpy(tmpl + n - 6, "XXXXXX", 6);
  }
  errno = EEXIST;
  return -1;
}

int getrusage(int who, struct rusage* usage) {
  (void)who;
  memset(usage, 0, sizeof(*usage));
  FILETIME create, exit_time, kernel, user;
  if (GetProcessTimes(GetCurrentProcess(), &create, &exit_time, &kernel, &user)) {
    auto to_tv = [](const FILETIME& ft, gufo_timeval_ru& tv) {
      const unsigned long long t100ns =
          (static_cast<unsigned long long>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
      tv.tv_sec = static_cast<long>(t100ns / 10000000ULL);
      tv.tv_usec = static_cast<long>((t100ns % 10000000ULL) / 10ULL);
    };
    to_tv(user, usage->ru_utime);
    to_tv(kernel, usage->ru_stime);
  }
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    usage->ru_maxrss = static_cast<long>(counters.PeakWorkingSetSize / 1024);
    usage->ru_majflt = static_cast<long>(counters.PageFaultCount);
  }
  return 0;
}

int uname(struct utsname* buf) {
  memset(buf, 0, sizeof(*buf));
  strcpy(buf->sysname, "Windows");
  DWORD size = sizeof(buf->nodename);
  if (!GetComputerNameA(buf->nodename, &size))
    strcpy(buf->nodename, "localhost");
  strcpy(buf->release, "10.0");
  strcpy(buf->version, "Windows");
  strcpy(buf->machine, "x86_64");
  return 0;
}

int kill(int pid, int sig) {
  (void)sig;
  HANDLE process = OpenProcess(PROCESS_TERMINATE, FALSE, static_cast<DWORD>(pid));
  if (process == nullptr) {
    errno = ESRCH;
    return -1;
  }
  const BOOL ok = TerminateProcess(process, 1);
  CloseHandle(process);
  return ok ? 0 : -1;
}

int waitpid(int pid, int* status, int options) {
  (void)pid; (void)options;
  if (status)
    *status = 0;
  errno = ECHILD;
  return -1;
}

int clock_gettime(clockid_t clk, struct timespec* ts) {
  if (clk == CLOCK_MONOTONIC) {
    static LARGE_INTEGER freq = [] {
      LARGE_INTEGER f;
      QueryPerformanceFrequency(&f);
      return f;
    }();
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    ts->tv_sec = static_cast<time_t>(now.QuadPart / freq.QuadPart);
    ts->tv_nsec = static_cast<long>((now.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
    return 0;
  }
  return timespec_get(ts, TIME_UTC) == TIME_UTC ? 0 : -1;
}

long syscall(long number, ...) {
  (void)number;
  errno = ENOSYS;
  return -1;
}

int posix_spawnp(pid_t*, const char*, const posix_spawn_file_actions_t*,
                 const posix_spawnattr_t*, char* const[], char* const[]) {
  return ENOSYS;
}
int posix_spawn(pid_t*, const char*, const posix_spawn_file_actions_t*,
                const posix_spawnattr_t*, char* const[], char* const[]) {
  return ENOSYS;
}
int posix_spawn_file_actions_init(posix_spawn_file_actions_t*) { return 0; }
int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t*) { return 0; }
int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t*, int, int) { return 0; }
int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t*, int) { return 0; }
int posix_spawn_file_actions_addopen(posix_spawn_file_actions_t*, int, const char*, int, int) {
  return 0;
}
int pipe(int fds[2]) {
  return _pipe(fds, 65536, _O_BINARY | _O_NOINHERIT);
}

int openat(int, const char*, int, ...) { errno = ENOSYS; return -1; }
int renameat(int, const char*, int, const char*) { errno = ENOSYS; return -1; }
int unlinkat(int, const char*, int) { errno = ENOSYS; return -1; }
int fstatat(int, const char*, struct _stat64*, int) { errno = ENOSYS; return -1; }
int utimensat(int, const char*, const struct timespec[2], int) { errno = ENOSYS; return -1; }
int geteuid(void) { return 0; }
int mincore(void*, size_t, unsigned char*) { errno = ENOSYS; return -1; }
int dprintf(int fd, const char* format, ...) {
  char buffer[4096];
  va_list args;
  va_start(args, format);
  const int n = vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (n <= 0)
    return n;
  return _write(fd, buffer, static_cast<unsigned>(n < (int)sizeof(buffer) ? n : (int)sizeof(buffer) - 1));
}
FILE* fmemopen(void*, size_t, const char*) { errno = ENOSYS; return nullptr; }
int sigaction(int sig, const struct sigaction* act, struct sigaction* old) {
  if (old) {
    old->sa_handler = SIG_DFL;
    old->sa_mask = 0;
    old->sa_flags = 0;
  }
  if (act && (sig == SIGINT || sig == SIGTERM || sig == SIGABRT))
    (void)signal(sig, act->sa_handler);
  return 0;
}
int getuid(void) { return 0; }

// ---- sockets -----------------------------------------------------------------
int gufo_socket_init(void) {
  static const int result = [] {
    WSADATA data;
    return WSAStartup(MAKEWORD(2, 2), &data);
  }();
  return result;
}

int gufo_socket_close(int fd) {
  return closesocket(static_cast<SOCKET>(fd)) == 0 ? 0 : -1;
}

int gufo_poll(struct pollfd* fds, unsigned long nfds, int timeout_ms) {
  // WSAPoll rejects output-only flags (POLLERR/POLLHUP/POLLNVAL) in events.
  for (unsigned long i = 0; i < nfds; ++i)
    fds[i].events &= (POLLRDNORM | POLLRDBAND | POLLWRNORM);
  const int result = WSAPoll(fds, nfds, timeout_ms);
  if (result < 0)
    errno = WSAGetLastError() == WSAEINTR ? EINTR : EIO;
  return result;
}

}  // extern "C"

#endif  // _WIN32
