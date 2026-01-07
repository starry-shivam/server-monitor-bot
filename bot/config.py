import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_IDS = [int(x) for x in os.getenv("OWNER_IDS", "0").split(",") if x.strip()]
LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID", "0"))
CALLBACK_TTL = int(os.getenv("CALLBACK_TTL", "300"))  # seconds
CALLBACK_SIG_SECRET = str(uuid.uuid4().hex)  # Unique per bot start
# Not a config, but useful to have here
BOT_START_TIME = time.time()

# --- Shell Config ---
SHELL_DENYLIST = {
    "sudo",
    "rm",
    "mv",
    "shutdown",
    "reboot",
    "init",
    "systemctl",
    "service",
    "bash",
    "zsh",
    "sh",
    "source",
    "pkill",
    "killall",
}
SHELL_TIMEOUT = int(os.getenv("SHELL_TIMEOUT", "45"))  # seconds
SHELL_MAX_OUTPUT = int(os.getenv("SHELL_MAX_OUTPUT", "3600"))

# --- Docker action config ---
DC_ALLOWED_ACTIONS = {"up", "stop", "pull", "build", "restart"}
DOCKER_APPS_DIR = Path(os.getenv("DOCKER_APPS_DIR", "/home/starry/docker-apps"))
DC_SCRIPT = os.getenv(
    "DC_SCRIPT", "dc_action.sh"
)  # Should be located in docker apps dir
DC_IGNORE_DIRS = set(os.getenv("DC_IGNORE_DIRS", "").split(","))
