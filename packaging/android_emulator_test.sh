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
# the app itself, watching the table the smoke test opened
adb shell input keyevent 4 >/dev/null 2>&1  # back (nothing to go back to: stays on the home page)
adb shell am start -n io.github.wlx7.guandan/.MainActivity >/dev/null 2>&1
sleep 2
adb exec-out screencap -p > emu/after.png
adb shell dumpsys activity services io.github.wlx7.guandan > emu/services.txt 2>&1
adb shell dumpsys notification --noredact 2>/dev/null | grep -A3 "io.github.wlx7.guandan" | head -20
grep -E "isForeground|foregroundServiceType" emu/services.txt | head -5
adb logcat -d > emu/logcat.txt
grep -iE "guandan|AndroidRuntime|FATAL|chromium" emu/logcat.txt | tail -40
# small copies of the screenshots in the log (artifacts are not always reachable)
python packaging/print_screenshot.py emu/home.png emu/after.png
exit $ok
