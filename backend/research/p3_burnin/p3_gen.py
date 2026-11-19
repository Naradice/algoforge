"""Generate one P3 ensemble: python p3_gen.py <cond> <seed_id>. Writes C:/tmp/p3_out/ens/<cond>_seed<k>.npz (+ .json info). Skips if present."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_env as E  # noqa: E402
import p3_sim as S  # noqa: E402


def path_for(cond, seed_id):
    return os.path.join(E.ENS_DIR, f"{cond}_seed{seed_id}.npz")


if __name__ == "__main__":
    cond, seed_id = sys.argv[1], int(sys.argv[2])
    out = path_for(cond, seed_id)
    if os.path.exists(out):
        print("exists", out)
        sys.exit(0)
    code = E.code_state()
    os.makedirs(E.ENS_DIR, exist_ok=True)
    t0 = time.time()
    ens, att, info = S.make_ensemble(cond, seed_id, workers=18, sep=(cond != "C0"))
    np.savez_compressed(out, ens=ens, attempt_ids=att)
    json.dump({"cond": cond, "seed_id": seed_id, "code": code, "info": info, "elapsed_s": time.time() - t0,
               "n_nan": int(np.isnan(ens).sum())}, open(out.replace(".npz", ".json"), "w"), indent=1)
    print("done", cond, seed_id, f"{time.time() - t0:.0f}s", "nan:", int(np.isnan(ens).sum()))
