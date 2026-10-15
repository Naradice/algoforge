export ARTIFACT_STORE_PATH=../artifacts
# wait for the 10D loop (Windows PID given as $1) to exit
while powershell -NoProfile -c "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 30; done
for i in 1 2 3 4 5 6; do
  timeout 7200 python phase10e_microstructure_screen.py 165 >> phase10e.log 2>&1 && break
  echo "retry $i after failure" >> phase10e.log; sleep 60
done
