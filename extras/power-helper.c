/*
 * Minimal privileged helper for reboot/poweroff.
 * Install as root:root with mode 4755 at /usr/local/bin/power-helper.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    ACTION_INVALID = -1,
};

static char *const REBOOT_ARGV[] = {
    "reboot",
    NULL,
};

static char *const POWEROFF_ARGV[] = {
    "poweroff",
    NULL,
};

static int run_action(const char *action, int *err) {
    if (strcmp(action, "reboot") == 0) {
        execv("/usr/sbin/reboot", REBOOT_ARGV);

        if (errno == ENOENT) {
            execv("/usr/bin/reboot", REBOOT_ARGV);
        }
        *err = errno;
        return 0;
    }

    if (strcmp(action, "poweroff") == 0) {
        execv("/usr/sbin/poweroff", POWEROFF_ARGV);

        if (errno == ENOENT) {
            execv("/usr/bin/poweroff", POWEROFF_ARGV);
        }
        *err = errno;
        return 0;
    }

    return ACTION_INVALID;
}

int main(int argc, char *argv[]) {
    int err;
    int action_result;

    if (argc != 2) {
        fprintf(stderr, "Usage: power-helper <reboot|poweroff>\n");
        return EXIT_FAILURE;
    }

    if (geteuid() != 0) {
        fprintf(stderr, "power-helper must be installed setuid root\n");
        return EXIT_FAILURE;
    }

    /* Drop inherited env vars before executing privileged binaries. */
    if (clearenv() != 0) {
        perror("clearenv");
        return EXIT_FAILURE;
    }

    err = 0;
    action_result = run_action(argv[1], &err);

    if (action_result == ACTION_INVALID) {
        fprintf(stderr, "invalid action\n");
        return EXIT_FAILURE;
    }

    errno = err;
    perror("execv");
    return (err == ENOENT) ? 127 : EXIT_FAILURE;
}
