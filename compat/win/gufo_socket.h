// Winsock mapping for the BSD socket calls Gufo's HTTP server uses.
#pragma once
#if defined(_WIN32)
#include "gufo_prelude.h"
#include <winsock2.h>
#include <ws2tcpip.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0
#endif
#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif
#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif
#ifndef SHUT_RD
#define SHUT_RD SD_RECEIVE
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH
#endif
typedef int socklen_t_gufo;

#ifdef __cplusplus
extern "C" {
#endif
// Starts Winsock once; safe to call many times.
int gufo_socket_init(void);
// close() for a socket descriptor.
int gufo_socket_close(int fd);
// poll() on sockets.
int gufo_poll(struct pollfd* fds, unsigned long nfds, int timeout_ms);
#ifdef __cplusplus
}
#endif
#define poll gufo_poll

#endif  // _WIN32
