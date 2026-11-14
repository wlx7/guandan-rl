/* LD_PRELOAD shim: the national-competition referee and its clients hard-code port 23456.
 * With ARENA_PORT set, bind()/connect() to port 23456 use ARENA_PORT instead, so that several
 * independent tables can run side by side on one machine. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

static void remap(const struct sockaddr *addr, socklen_t len, struct sockaddr_storage *out) {
    memcpy(out, addr, len < sizeof(*out) ? len : sizeof(*out));
    const char *p = getenv("ARENA_PORT");
    if (!p) return;
    unsigned short port = (unsigned short)atoi(p);
    if (addr->sa_family == AF_INET && ntohs(((struct sockaddr_in *)out)->sin_port) == 23456)
        ((struct sockaddr_in *)out)->sin_port = htons(port);
    else if (addr->sa_family == AF_INET6 && ntohs(((struct sockaddr_in6 *)out)->sin6_port) == 23456)
        ((struct sockaddr_in6 *)out)->sin6_port = htons(port);
}

int bind(int fd, const struct sockaddr *addr, socklen_t len) {
    static int (*real)(int, const struct sockaddr *, socklen_t);
    if (!real) real = dlsym(RTLD_NEXT, "bind");
    struct sockaddr_storage s;
    remap(addr, len, &s);
    return real(fd, (struct sockaddr *)&s, len);
}

int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    static int (*real)(int, const struct sockaddr *, socklen_t);
    if (!real) real = dlsym(RTLD_NEXT, "connect");
    struct sockaddr_storage s;
    remap(addr, len, &s);
    return real(fd, (struct sockaddr *)&s, len);
}
