#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdnoreturn.h>
#include <errno.h>

struct string {
    char *str; // not null-terminated
    int len;
};

void printInt(int n) {
    printf("%d\n", n);
}

void printString(struct string *s) {
    printf("%.*s\n", s->len, s->str);
}

noreturn void error() {
    printf("runtime error\n");
    exit(1);
}

int readInt() {
    // use getline first
    int n;
    char *buf = NULL;
    size_t len = 0;
    ssize_t nread = getline(&buf, &len, stdin);
    if (nread < 0) {
        error();
    } else {
        //use strtol to convert the string to an integer
        char *endptr;
        errno = 0;
        n = strtol(buf, &endptr, 10);
        if (endptr == buf || *endptr != '\n' || errno != 0) {
            error();
        }
        free(buf);
        return n;
    }
}

struct string *readString() {
    // use getline to read a line of input and allocate a buffer for it
    char *buf = NULL;
    size_t len = 0;
    ssize_t nread = getline(&buf, &len, stdin);
    if (nread < 0) {
        error();
    } else {
        // remove the newline character at the end
        buf[nread - 1] = '\0';
        struct string *s = malloc(sizeof(struct string));
        s->str = buf;
        s->len = nread - 1;
        return s;
    }
}

struct string* newString(char* str, int len) {
    struct string *s = malloc(sizeof(struct string));
    char *new_str = malloc(len);
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
    char* new_str = malloc(new_len);
    verifyNotNull(new_str);
    s->str = new_str;
    memcpy(new_str, s1->str, s1->len);
    memcpy(new_str + s1->len, s2->str, s2->len);
    return s;
}
