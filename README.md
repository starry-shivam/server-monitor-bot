# Server Monitor Bot

Telegram bot for server monitoring and managing my homeserver. **Intended for personal use only, No issues/PR accepted**.

## What It Does

- Shows system and container status
- Runs restricted maintenance actions
- Supports scheduled background jobs
- Loads features dynamically, so modules can be enabled or disabled cleanly

## Commands

Core:

- `/start` Start the bot
- `/help` Show available commands
- `/ping` Check Telegram API latency and host uptime

Feature commands:

- `/fetch` Fast system summary
- `/metrics` CPU, RAM, and disk visual stats
- `/dockerps` Docker container status
- `/dcpanel` Quick Docker app control panel
- `/dcaction` Docker Compose operations
- `/dcupdate` Check container image updates
- `/shell` Run allowlisted read-only shell commands (optional)
- `/dragon` Dragon Q6A hardware report (only on matching host)
- `/reboot` Reboot host (optional)
- `/poweroff` Power off host (optional)
- `/pyexec` Execute Python snippets (optional)
- `/update_playlist` Navidrome playlist rebuild + scan (optional, personal workflow module)

## Quick Setup

1. Install dependencies:

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` with at least:

```dotenv
BOT_TOKEN=your_bot_token
OWNER_IDS=123456789
CALLBACK_SIG_SECRET=change_me
```

3. Optional feature flags:

```dotenv
POWER_MGMT_AVAILABLE=true
SHELL_ENABLED=false
PYEXEC_ENABLED=false
NOTIFY_DOCKER_UPDATES=true
```

4. Run:

```bash
python -m bot
```

## Run As systemd Service

Use the unit file at `extras/server-monitor-bot.service`.

1. Copy unit file:

```bash
sudo cp extras/server-monitor-bot.service /etc/systemd/system/server-monitor-bot.service
```

2. Edit paths/user in unit file to match your host:

- `WorkingDirectory`
- `ExecStart`
- `User`
- `ReadWritePaths` (if needed)

3. Reload and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now server-monitor-bot.service
```

4. Check status and logs:

```bash
systemctl status server-monitor-bot.service
journalctl -u server-monitor-bot.service -f
```

5. After bot updates:

```bash
sudo systemctl restart server-monitor-bot.service
```

## Extra Feature: Power Management (Setup)

To load `/reboot` and `/poweroff` commands, enable power management in `.env`:

```dotenv
POWER_MGMT_AVAILABLE=true
```

The `/reboot` command is rate-limited for the first 3 minutes after boot.

To load `/shell`, enable it explicitly in `.env`:

```dotenv
SHELL_ENABLED=true
```

After changing `.env`, restart the bot (or restart the systemd service).

Power actions use a tiny setuid C helper at `/usr/local/bin/power-helper`.

1. Verify command paths:

```bash
command -v reboot
command -v poweroff
```

Expected on Debian 13:

```text
/usr/sbin/reboot
/usr/sbin/poweroff
```

2. Build and install:

```bash
gcc -O2 -Wall -Wextra -Wpedantic -Werror -D_FORTIFY_SOURCE=3 -fstack-protector-strong -fPIE -pie -Wl,-z,relro,-z,now -o /tmp/power-helper extras/power-helper.c
sudo install -o root -g root -m 4755 /tmp/power-helper /usr/local/bin/power-helper
```

3. Verify ownership and mode:

```bash
ls -l /usr/local/bin/power-helper
```

Expected:

```text
-rwsr-xr-x 1 root root ... /usr/local/bin/power-helper
```
