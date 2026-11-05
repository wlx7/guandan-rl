#!/bin/bash
# CI: install the APK on the emulator, start it, and play an AI table through its server.
set -x
mkdir -p emu
adb install -r guandan_android.apk || exit 1
adb shell pm grant io.github.wlx7.guandan android.permission.POST_NOTIFICATIONS 2>/dev/null
adb shell am start -W -n io.github.wlx7.guandan/.MainActivity
sleep 12
adb forward tcp:8000 tcp:8000
adb exec-out screencap -p > emu/home.png
python packaging/ci_smoke.py 8000
ok=$?
# watch the table in the app itself
adb shell am start -a android.intent.action.VIEW -n io.github.wlx7.guandan/.MainActivity >/dev/null 2>&1
sleep 3
adb exec-out screencap -p > emu/after.png
adb shell dumpsys activity services io.github.wlx7.guandan > emu/services.txt 2>&1
adb logcat -d > emu/logcat.txt
grep -iE "guandan|AndroidRuntime|FATAL" emu/logcat.txt | tail -60
exit $ok
