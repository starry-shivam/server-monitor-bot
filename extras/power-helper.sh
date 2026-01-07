#!/bin/bash

case "$1" in
  reboot)
    /usr/bin/systemctl reboot
    ;;
  poweroff)
    /usr/bin/systemctl poweroff
    ;;
  *)
    exit 1
esac
