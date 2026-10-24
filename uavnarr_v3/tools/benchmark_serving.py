import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def worker(gpu, run, warm, task_q, result_q, ready_q, prefetch=0, cpu_workers=4):
    gpu_id = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, ROOT)
    import torch
    from uavnarr import data
    from uavnarr.model import import_videollama2, load_base

    videollama2, *_ = import_videollama2()
    model, processor, tokenizer = load_base()
    from infer import load_trained
    from uavnarr.sample import SampleBuilder
    model, cfg = load_trained(model, run)
    model = model.to("cuda").eval()
    builder = SampleBuilder(cfg, processor)

    def prep(v):
        t0 = time.perf_counter()
        attrs = data.load_attributes(v)
        video, prompt, side = builder.build(v, attrs=attrs)
        return {"attrs": attrs, "video": video, "prompt": prompt, "side": side,
                "load_ms": 1e3 * (time.perf_counter() - t0)}

    def gpu(v, p):
        side = p["side"]
        if "zoom" in side:
            side["zoom"] = side["zoom"].to("cuda", torch.float16)
        if getattr(model, "uav_attr_cond", None) is not None:
            vec = ([1.0] * data.ENCODING_DIM[cfg["attr_encoding"]] if cfg.get("attr_constant")
                   else data.encode_attributes(p["attrs"], cfg["attr_encoding"]))
            model._attr_vec = torch.tensor(vec, device="cuda")
        model._side = side
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        cap = videollama2.mm_infer(p["video"], p["prompt"], model=model, tokenizer=tokenizer,
                                   do_sample=False, max_new_tokens=100).strip()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return cap, 1e3 * (t3 - t2)

    def one(v):
        t0 = time.perf_counter()
        p = prep(v)
        cap, cap_ms = gpu(v, p)
        return {"video": v, "gpu": gpu_id, "load_ms": p["load_ms"], "caption_ms": cap_ms,
                "total_ms": 1e3 * (time.perf_counter() - t0), "words": len(cap.split()), "caption": cap}

    for v in warm:
        one(v)
    torch.cuda.reset_peak_memory_stats()
    ready_q.put({"gpu": gpu_id, "name": torch.cuda.get_device_name(0),
                 "mem_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
    if not prefetch:
        while True:
            v = task_q.get()
            if v is None:
                break
            try:
                result_q.put(one(v))
            except Exception as e:
                result_q.put({"video": v, "gpu": gpu_id, "error": repr(e)})
    else:
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(cpu_workers)
        pending = queue.Queue(maxsize=prefetch)

        def feeder():
            while True:
                v = task_q.get()
                if v is None:
                    pending.put((None, None))
                    return
                pending.put((v, pool.submit(prep, v)))

        threading.Thread(target=feeder, daemon=True).start()
        while True:
            v, fut = pending.get()
            if v is None:
                break
            try:
                t0 = time.perf_counter()
                p = fut.result()
                wait_ms = 1e3 * (time.perf_counter() - t0)
                cap, cap_ms = gpu(v, p)
                result_q.put({"video": v, "gpu": gpu_id, "load_ms": p["load_ms"], "wait_ms": wait_ms,
                              "caption_ms": cap_ms, "total_ms": wait_ms + cap_ms,
                              "words": len(cap.split()), "caption": cap})
            except Exception as e:
                result_q.put({"video": v, "gpu": gpu_id, "error": repr(e)})
        pool.shutdown()
    result_q.put({"done": True, "gpu": gpu_id, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})

def environment():
    env = {"host": platform.node(), "python": platform.python_version(), "cpu_cores": os.cpu_count()}
    try:
        env["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if "model name" in l)
        env["ram_gb"] = round(int(next(l.split()[1] for l in open("/proc/meminfo")
                                       if l.startswith("MemTotal"))) / 1e6, 1)
    except Exception:
        pass
    try:
        env["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    except Exception:
        pass
    try:
        import torch, transformers, peft
        env.update(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                   peft=peft.__version__)
    except Exception:
        pass
    return env

def stats(x):
    x = np.asarray(x, dtype=float)
    return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "mean": float(x.mean())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trained run, e.g. runs/attrcond/seed0")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N videos")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="0 = sequential (CPU stage then GPU stage); N > 0 = pipelined, up to N videos "
                         "prepared ahead by --cpu-workers threads while the GPU captions")
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from uavnarr import data
    vids = [v for s in a.splits.split(",") for v in data.load_split(s)]
    if a.limit:
        vids = vids[:a.limit]
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    warm = vids[:a.warmup]
    print(f"{len(vids)} videos on {len(gpus)} GPU(s) {gpus}")

    ctx = mp.get_context("spawn")
    task_q, result_q, ready_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(g, a.run, warm, task_q, result_q, ready_q, a.prefetch, a.cpu_workers)) for g in gpus]
    for p in procs:
        p.start()
    devices = [ready_q.get() for _ in gpus]
    t0 = time.perf_counter()
    for v in vids:
        task_q.put(v)
    for _ in gpus:
        task_q.put(None)
    recs, peak, done = [], {}, 0
    while done < len(gpus):
        r = result_q.get()
        if r.get("done"):
            peak[r["gpu"]] = r["peak_mem_gb"]
            done += 1
        else:
            recs.append(r)
            if len(recs) % 100 == 0:
                print(f"  {len(recs)}/{len(vids)}  {time.perf_counter() - t0:.0f}s")
    wall = time.perf_counter() - t0
    for p in procs:
        p.join()

    ok = [r for r in recs if "error" not in r]
    err = [r for r in recs if "error" in r]
    summ = {"environment": environment(), "devices": devices, "run": a.run, "n_gpus": len(gpus),
            "videos": len(vids), "ok": len(ok), "failed": len(err), "wall_time_s": wall,
            "videos_per_hour": 3600 * len(ok) / wall,
            "videos_per_gpu_hour": 3600 * len(ok) / wall / len(gpus),
            "peak_mem_gb_per_replica": peak,
            "mode": f"pipelined (prefetch {a.prefetch}, {a.cpu_workers} CPU threads)" if a.prefetch else "sequential",
            "latency_ms": {k: stats([r[k] for r in ok]) for k in ("load_ms", "wait_ms", "caption_ms", "total_ms")
                           if ok and k in ok[0]},
            "words_mean": float(np.mean([r["words"] for r in ok])) if ok else None}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": summ, "records": recs}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    if err:
        print(f"WARNING: {len(err)} videos failed, e.g. {err[:2]}")

if __name__ == "__main__":
    main()
