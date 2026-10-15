until ! powershell -c "Get-Process -Id 32092 -ErrorAction Stop" >/dev/null 2>&1; do sleep 20; done
ARTIFACT_STORE_PATH=../artifacts timeout 10800 python phase10a_target_screen.py 29 >> phase10a_29.log 2>&1
