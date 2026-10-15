export ARTIFACT_STORE_PATH=../artifacts
timeout 5400 python phase10a_target_screen.py 149 >> phase10a_149.log 2>&1
timeout 10800 python phase10a_target_screen.py 29 >> phase10a_29.log 2>&1
