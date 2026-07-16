/*
 * Minimal privileged helper for reboot/poweroff.
 * Install as root:root with mode 4755 at /usr/local/bin/power-helper.
 */

#include <errno.h>
#include <grp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    ACTION_EXEC_FAILED = 0,
    ACTION_INVALID = -1,
};

static char *const REBOOT_ARGV[] = {"reboot", NULL};
static char *const POWEROFF_ARGV[] = {"poweroff", NULL};

static int run_action(const char *action) {
    if (strcmp(action, "reboot") == 0) {
        if (execv("/usr/sbin/reboot", REBOOT_ARGV) == -1 &&
            errno == ENOENT) {
            execv("/usr/bin/reboot", REBOOT_ARGV);
        }
        return ACTION_EXEC_FAILED;
    }

    if (strcmp(action, "poweroff") == 0) {
        if (execv("/usr/sbin/poweroff", POWEROFF_ARGV) == -1 &&
            errno == ENOENT) {
            execv("/usr/bin/poweroff", POWEROFF_ARGV);
        }
        return ACTION_EXEC_FAILED;
    }

    return ACTION_INVALID;
}

int main(int argc, char *argv[]) {
    int action_result;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <reboot|poweroff>\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (geteuid() != 0) {
        fprintf(stderr, "power-helper must be installed setuid root\n");
        return EXIT_FAILURE;
    }

    if (setgroups(0, NULL) != 0) {
        perror("setgroups");
        return EXIT_FAILURE;
    }

    if (setgid(0) != 0) {
        perror("setgid");
        return EXIT_FAILURE;
    }

    if (setuid(0) != 0) {
        perror("setuid");
        return EXIT_FAILURE;
    }

    /* Drop inherited env vars before executing privileged binaries. */
    if (clearenv() != 0) {
        perror("clearenv");
        return EXIT_FAILURE;
    }

    action_result = run_action(argv[1]);

    if (action_result == ACTION_INVALID) {
        fprintf(stderr, "invalid action\n");
        return EXIT_FAILURE;
    }

    perror("exec");
    return (errno == ENOENT) ? 127 : EXIT_FAILURE;
}
