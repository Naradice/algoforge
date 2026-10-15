export ARTIFACT_STORE_PATH=../artifacts
set -a; . ./.env; set +a
for i in 1 2 3 4; do
  timeout 14400 python phase11_activity_scratch.py baselines >> phase11_baselines.log 2>&1 && break
  echo "retry $i after failure" >> phase11_baselines.log; sleep 120
done
echo FINISHED >> phase11_baselines.log
