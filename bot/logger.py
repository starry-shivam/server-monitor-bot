#!/usr/bin/env python3

import logging


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _user_label(user) -> str:
    if not user:
        return "user=<unknown>"
    username = getattr(user, "username", None)
    username_part = f" @{username}" if username else ""
    return f"user={getattr(user, 'id', '?')}{username_part}"


def _chat_label(chat) -> str:
    if not chat:
        return "chat=<unknown>"
    title = getattr(chat, "title", None) or getattr(chat, "type", None) or "unknown"
    return f"chat={getattr(chat, 'id', '?')} ({title})"


def log_command(logger: logging.Logger, update, command_name: str, status: str, detail: str = "") -> None:
    message = f"command={command_name} status={status} {_user_label(getattr(update, 'effective_user', None))} {_chat_label(getattr(update, 'effective_chat', None))}"
    if detail:
        message = f"{message} detail={detail}"
    logger.info(message)


def log_callback(logger: logging.Logger, user, source: str, action: str, status: str, detail: str = "") -> None:
    message = f"callback={source} action={action} status={status} {_user_label(user)}"
    if detail:
        message = f"{message} detail={detail}"
    logger.info(message)


def log_job(logger: logging.Logger, job_name: str, status: str, detail: str = "") -> None:
    message = f"job={job_name} status={status}"
    if detail:
        message = f"{message} detail={detail}"
    logger.info(message)


def log_security_event(logger: logging.Logger, event: str, status: str, detail: str = "") -> None:
    message = f"security_event={event} status={status}"
    if detail:
        message = f"{message} detail={detail}"
    logger.warning(message)