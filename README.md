# Server Monitor Bot 🐧

Lightweight Telegram bot to monitor and manage a Linux server. Provides system info, live stats charts, shell/python exec utilities and update/restart integration via systemd.

## Commands
(All commands except `/start` are owner-restricted, set `OWNER_ID` to your numeric Telegram id.)

- `/start` — Welcome message
- `/help` — Show available commands (owner-only)
- `/info` — System info (neofetch-like)
- `/info -ip` — System info + public IP
- `/cputemp` — CPU temperature on supported hardware (e.g., Raspberry Pi).
- `/stats` — Static chart of CPU / RAM / Disk
- `/stats -live` — Live monitoring (10 updates)
- `/ping` — Measure Telegram API latency and uptime
- `/ip` — Show server public IP
- `/shell` <cmd...> — Run shell command on server
- `/pyexec` <python code> — Execute Python snippet remotely
- `/updatebot` — Git pull and restart systemd service

## Prerequisites
- Python 3.9+ (3.10 recommended)
- System packages for building wheels if needed (build-essential, libffi-dev, etc.)
- A Telegram bot token and your Telegram numeric user id

## Install & run (local)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export BOT_TOKEN="123:ABC..."
export OWNER_ID="123456789"
export OWNER_USERNAME="myuser"
python3 bot.py
```

## Run as a systemd service (recommended)
1. Copy and edit the template file `server-monitor-bot/server-monitor-bot.service`
2. Set `User`,`WorkingDirectory`, `ExecStart`, and environment vars (`BOT_TOKEN`, `OWNER_ID`, `OWNER_USERNAME`).
3. Install and enable:

```bash
sudo cp server-monitor-bot.service /etc/systemd/system/server-monitor-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now server-monitor-bot.service
sudo systemctl status server-monitor-bot.service
```

To view logs:
```bash
journalctl -u server-monitor-bot.service -f
```

## Security notes
- `/shell` and `/pyexec` allow arbitrary code execution — use only in trusted environments.
- Owner-only decorator enforces `OWNER_ID`; ensure `OWNER_ID` is set correctly.
- Review and adapt the systemd unit for additional hardening as needed.

## Example Output

<img width="1543" height="1286" alt="{2B93CAB9-2D01-41C4-8FC4-A145D833BBCF}" src="https://github.com/user-attachments/assets/c4dfdd7b-f7a5-41cc-a652-36ee24acf7fd" />

## License

```
MIT License

Copyright (c) [2025 - Present] Stɑrry Shivɑm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
