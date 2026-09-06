#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdnoreturn.h>
#include <errno.h>
#include <unistd.h>

struct string {
    char *str; // not null-terminated
    int len;
};

void verifyNotNull(void *p);
void verifyString(struct string *s);

void printInt(int n) {
    printf("%d\n", n);
}

void printString(struct string *s) {
    // Flush earlier buffered `printf` output so file-captured streams stay
    // ordered relative to these length-based direct writes.
    fflush(stdout);
    write(STDOUT_FILENO, s->str, s->len);
    write(STDOUT_FILENO, "\n", 1);
}

noreturn void error() {
    printf("runtime error\n");
    exit(1);
}

int readInt() {
    char *buf = NULL;
    size_t len = 0;
    ssize_t nread = getline(&buf, &len, stdin);
    if (nread < 0) {
        error();
    } else {
        char *endptr;
        errno = 0;
        long n = strtol(buf, &endptr, 10);
        if (endptr == buf || errno != 0) {
            error();
        }
        if (n < -2147483648L || n > 2147483647L) {
            error();
        }
        // Allow trailing whitespace, including a single newline, CRLF,
        // or EOF-terminated input without a newline.
        while (*endptr == ' ' || *endptr == '\t' || *endptr == '\r' || *endptr == '\n') {
            endptr++;
        }
        if (*endptr != '\0') {
            error();
        }
        int result = (int)n;
        free(buf);
        return result;
    }
}

struct string *readString() {
    char *buf = NULL;
    size_t len = 0;
    ssize_t nread = getline(&buf, &len, stdin);
    if (nread < 0) {
        error();
    } else {
        // Strip a single trailing newline (and preceding CR for CRLF);
        // EOF-terminated input without a newline keeps its last byte.
        ssize_t content_len = nread;
        if (content_len > 0 && buf[content_len - 1] == '\n') {
            content_len--;
        }
        if (content_len > 0 && buf[content_len - 1] == '\r') {
            content_len--;
        }
        buf[content_len] = '\0';
        struct string *s = malloc(sizeof(struct string));
        verifyNotNull(s);
        s->str = buf;
        s->len = (int)content_len;
        return s;
    }
}

struct string* newString(char* str, int len) {
    struct string *s = malloc(sizeof(struct string));
    verifyNotNull(s);
    char *new_str = malloc(len > 0 ? len : 1);
    verifyNotNull(new_str);
    memcpy(new_str, str, len);
    s->str = new_str;
    s->len = len;
    return s;
}

int stringEqual(struct string *s1, struct string *s2) {
    return s1->len == s2->len && memcmp(s1->str, s2->str, s1->len) == 0;
}

void verifyNotNull(void *p) {
    if (p == NULL) {
        error();
    }
}

void verifyString(struct string *s) {
    verifyNotNull(s);
    verifyNotNull(s->str);
}

struct string *stringConcat(struct string *s1, struct string *s2) {
    verifyString(s1);
    verifyString(s2);
    struct string *s = malloc(sizeof(struct string));
    verifyNotNull(s);
    int new_len = s1->len + s2->len;
    s->len = new_len;
    char* new_str = malloc(new_len > 0 ? new_len : 1);
    verifyNotNull(new_str);
    s->str = new_str;
    memcpy(new_str, s1->str, s1->len);
    memcpy(new_str + s1->len, s2->str, s2->len);
    return s;
}
