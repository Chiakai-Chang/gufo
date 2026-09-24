// Small socket helpers shared by the HTTP and WebSocket servers.
// POSIX and Winsock differ in close(), timeouts, SIGPIPE and startup.
#pragma once

#include <sys/socket.h>
#ifndef _WIN32
#include <csignal>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace gufo::platform {

inline void IgnoreSigpipe() {
#ifndef _WIN32
  (void)::signal(SIGPIPE, SIG_IGN);
#endif
}

inline int OpenTcpSocket() {
#ifdef _WIN32
  if (gufo_socket_init() != 0)
    return -1;
  const SOCKET s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  return s == INVALID_SOCKET ? -1 : static_cast<int>(s);
#else
  return ::socket(AF_INET, SOCK_STREAM, 0);
#endif
}

inline int AcceptSocket(int listen_fd) {
#ifdef _WIN32
  const SOCKET s = ::accept(static_cast<SOCKET>(listen_fd), nullptr, nullptr);
  return s == INVALID_SOCKET ? -1 : static_cast<int>(s);
#else
  return ::accept(listen_fd, nullptr, nullptr);
#endif
}

inline int CloseSocket(int fd) {
#ifdef _WIN32
  return gufo_socket_close(fd);
#else
  return ::close(fd);
#endif
}

inline void SetSocketTimeouts(int fd, int seconds, bool receive, bool send) {
#ifdef _WIN32
  const DWORD ms = static_cast<DWORD>(seconds) * 1000u;
  const char* value = reinterpret_cast<const char*>(&ms);
  const int size = sizeof(ms);
#else
  const timeval tv{seconds, 0};
  const void* value = &tv;
  const socklen_t size = sizeof(tv);
#endif
  if (receive)
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, value, size);
  if (send)
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, value, size);
}

}  // namespace gufo::platform
