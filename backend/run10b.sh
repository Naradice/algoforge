for i in 1 2 3 4 5 6; do
  timeout 7200 python phase10b_cross_pair_screen.py >> phase10b.log 2>&1 && break
  echo "retry $i after failure" >> phase10b.log; sleep 60
done
