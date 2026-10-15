export ARTIFACT_STORE_PATH=../artifacts
for i in 1 2 3 4 5 6; do
  timeout 7200 python phase10d_volume_screen.py 158 >> phase10d.log 2>&1 && break
  echo "retry $i after failure" >> phase10d.log; sleep 60
done
