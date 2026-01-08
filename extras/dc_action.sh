#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025-Present Stɑrry Shivɑm <starry@krsh.dev>
# All Rights Reserved. // This file is a part of server-monitor-bot
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------
# 1. Configuration & Colors
# -----------------------------

NO_COLOR=0

# Colors (default enabled)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Allowed actions whitelist
VALID_ACTIONS=("up" "down" "start" "stop" "restart" "pause" "unpause" "pull" "ps" "logs")

# -----------------------------
# 2. Helper Functions
# -----------------------------

print_usage() {
    echo -e "${YELLOW}Usage:${NC} $0 <action> [--ignore dir1,dir2] [--no-color]"
    echo -e "${YELLOW}Valid actions:${NC} ${VALID_ACTIONS[*]}"
    exit 1
}

# Wrapper to handle 'docker compose' vs 'docker-compose'
run_compose() {
    if docker compose version &>/dev/null; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

# -----------------------------
# 3. Parse arguments
# -----------------------------

[[ -z "$1" ]] && print_usage

ACTION="$1"
shift

MATCH=0
for act in "${VALID_ACTIONS[@]}"; do
    [[ "$ACTION" == "$act" ]] && MATCH=1 && break
done

if [[ "$MATCH" -eq 0 ]]; then
    echo -e "${RED}❌ Error: Unsupported action '$ACTION'.${NC}"
    print_usage
fi

IGNORE_DIRS=() # default ignore

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ignore)
            IFS=',' read -r -a NEW_IGNORES <<< "$2"
            IGNORE_DIRS+=("${NEW_IGNORES[@]}")
            shift 2
            ;;
        --no-color)
            NO_COLOR=1
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Disable colors if requested
if [[ "$NO_COLOR" -eq 1 ]]; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# -----------------------------
# 4. Main Loop
# -----------------------------

SUMMARY_LOG=()

echo -e "${BLUE}🚀 Starting bulk operation: ${ACTION}${NC}"
echo "----------------------------------------"

for d in */ ; do
    DIR="${d%/}"

    # Ignore logic
    SKIP=0
    for ignore in "${IGNORE_DIRS[@]}"; do
        if [[ "$DIR" == "$ignore" ]]; then
            echo -e "${YELLOW}⏭  Skipping $DIR (ignored)${NC}"
            SUMMARY_LOG+=("${DIR}:${YELLOW}Ignored${NC}")
            SKIP=1
            break
        fi
    done
    [[ "$SKIP" -eq 1 ]] && continue

    # Check for compose files
    if [[ -f "$d/docker-compose.yml" || -f "$d/docker-compose.yaml" || \
          -f "$d/compose.yml" || -f "$d/compose.yaml" ]]; then

        echo -e "${CYAN}➡ Entering $DIR${NC}"

        (
            cd "$d" || exit 1

            RUNNING_COUNT=$(run_compose ps -q --filter "status=running" 2>/dev/null | wc -l)
            STOPPED_COUNT=$(run_compose ps -q --filter "status=exited" 2>/dev/null | wc -l)
            PAUSED_COUNT=$(run_compose ps -q --filter "status=paused" 2>/dev/null | wc -l)
            TOTAL_COUNT=$(run_compose ps -q 2>/dev/null | wc -l)

            # -----------------------------
            # Smart Action Guards
            # -----------------------------

            # up
            if [[ "$ACTION" == "up" && "$RUNNING_COUNT" -gt 0 ]]; then
                echo -e "   ${GREEN}✔ Already running. Skipping 'up'.${NC}"
                exit 10
            fi

            # start
            if [[ "$ACTION" == "start" ]]; then
                if [[ "$RUNNING_COUNT" -gt 0 ]]; then
                    echo -e "   ${GREEN}✔ Already running. Skipping 'start'.${NC}"
                    exit 10
                fi
                if [[ "$TOTAL_COUNT" -eq 0 ]]; then
                    echo -e "   ${YELLOW}⚠ No containers exist. Skipping 'start'.${NC}"
                    exit 11
                fi
            fi

            # stop / down
            if [[ "$ACTION" == "stop" || "$ACTION" == "down" ]]; then
                if [[ "$RUNNING_COUNT" -eq 0 ]]; then
                    echo -e "   ${GREEN}✔ Already stopped. Skipping '$ACTION'.${NC}"
                    exit 10
                fi
            fi

            # restart
            if [[ "$ACTION" == "restart" && "$RUNNING_COUNT" -eq 0 ]]; then
                echo -e "   ${YELLOW}⚠ Not running. Skipping 'restart'.${NC}"
                exit 11
            fi

            # pause
            if [[ "$ACTION" == "pause" ]]; then
                if [[ "$RUNNING_COUNT" -eq 0 || "$PAUSED_COUNT" -gt 0 ]]; then
                    echo -e "   ${GREEN}✔ Nothing to pause. Skipping 'pause'.${NC}"
                    exit 10
                fi
            fi

            # unpause
            if [[ "$ACTION" == "unpause" && "$PAUSED_COUNT" -eq 0 ]]; then
                echo -e "   ${GREEN}✔ Nothing paused. Skipping 'unpause'.${NC}"
                exit 10
            fi

            # -----------------------------
            # Execution
            # -----------------------------

            if [[ "$ACTION" == "up" ]]; then
                echo -e "   Running: ${GREEN}docker compose up -d${NC}"
                run_compose up -d --no-build
            else
                echo -e "   Running: ${GREEN}docker compose $ACTION${NC}"
                run_compose "$ACTION"
            fi
        )

        EXIT_CODE=$?

        if [[ "$EXIT_CODE" -eq 0 ]]; then
            SUMMARY_LOG+=("${DIR}:${GREEN}Success${NC}")
        elif [[ "$EXIT_CODE" -eq 10 ]]; then
            SUMMARY_LOG+=("${DIR}:${GREEN}Skipped${NC}")
        elif [[ "$EXIT_CODE" -eq 11 ]]; then
            SUMMARY_LOG+=("${DIR}:${YELLOW}Skipped (N/A)${NC}")
        else
            SUMMARY_LOG+=("${DIR}:${RED}Failed${NC}")
        fi

    else
        echo -e "   ${YELLOW}No compose file found — skipping.${NC}"
        SUMMARY_LOG+=("${DIR}:${YELLOW}No Compose File${NC}")
    fi
done

# -----------------------------
# 5. Summary Output
# -----------------------------

echo ""
echo "----------------------------------------"
echo -e "${BLUE}📊 Execution Summary${NC}"
echo "----------------------------------------"
printf "%-30s %s\n" "DIRECTORY" "STATUS"
echo "----------------------------------------"

for entry in "${SUMMARY_LOG[@]}"; do
    DIR_NAME="${entry%%:*}"
    STATUS="${entry#*:}"
    printf "%-30s %b\n" "$DIR_NAME" "$STATUS"
done

echo ""
echo -e "${GREEN}✅ All done!${NC}"
