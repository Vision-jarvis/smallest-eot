"""Download smart-turn-data-v3.2 shards one at a time, keep Hindi rows (and, from shard 11
on, real-human non-synthetic rows of every language), delete the shard."""
import os, sys, time, pyarrow.parquet as pq, pyarrow.compute as pc
from huggingface_hub import hf_hub_download

repo = sys.argv[1]  # smart-turn-data-v3.2-train | smart-turn-data-v3.2-test
n = int(sys.argv[2])
dst = "data/smart_turn_hi" if repo.endswith("train") else "data/smart_turn_hi_test"
os.makedirs(dst, exist_ok=True)
for i in range(n):
    out = f"{dst}/hi_{i:05d}.parquet"
    hum = f"{dst}/human_{i:05d}.parquet"
    if os.path.exists(out) and (os.path.exists(hum) or i <= 10):
        continue
    t0 = time.time()
    for attempt in range(5):
        try:
            p = hf_hub_download(f"pipecat-ai/{repo}", f"data/train-{i:05d}-of-{n:05d}.parquet", repo_type="dataset", local_dir=f"data/_tmp_{repo}")
            break
        except Exception as e:  # flaky network: retry
            print("retry", i, type(e).__name__, flush=True)
            time.sleep(10)
    t = pq.read_table(p)
    if not os.path.exists(out):
        pq.write_table(t.filter(pc.equal(t["language"], "hin")), out)
    h = t.filter(pc.and_(pc.invert(t["synthetic"]), pc.not_equal(t["language"], "hin")))
    pq.write_table(h, hum)
    os.remove(p)
    print(i, f"human={h.num_rows}", f"{time.time()-t0:.0f}s", flush=True)
