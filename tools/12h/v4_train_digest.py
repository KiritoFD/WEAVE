import json

rows = [json.loads(l) for l in open("runs/v4/prior/train.jsonl")]
val = [r for r in rows if r.get("event") == "val"]
tr = [r for r in rows if r.get("event") == "train"]
print("val points:", len(val), "train points:", len(tr))
for s in (250, 1000, 3000, 6000, 9250, 10000, 10500):
    hit = [r for r in val if r["step"] == s]
    if hit:
        r = hit[0]
        print(f"val step={s} loss={r['val_loss']:.4f} buckets={[round(x,3) for x in r.get('buckets', [])]}")
for s in (1000, 3000, 6000, 10000, 10500):
    hit = [r for r in tr if r["step"] == s]
    if hit:
        r = hit[0]
        print(f"train step={s} loss={r['loss']:.4f} lr={r['lr']:.2e}")
spans = [(tr[i]["step"], tr[i]["elapsed_s"]) for i in range(0, len(tr), 50)]
if len(spans) >= 2:
    (s0, t0), (s1, t1) = spans[0], spans[-1]
    print(f"step time = {(t1 - t0) / (s1 - s0):.3f} s/step  (mb=16 x accum 2)")

