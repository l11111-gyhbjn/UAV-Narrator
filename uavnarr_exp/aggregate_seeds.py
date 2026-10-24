import argparse
import glob
import statistics

from common import ATTRS, load_json, save_json

def fmt(m, s, digits=2, std=True):
    if std and s is not None:
        return f"{m:.{digits}f}{{\\scriptsize$\\pm${s:.{digits}f}}}"
    return f"{m:.{digits}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--metrics", nargs="+",
                    default=["B1", "B2", "B3", "B4", "METEOR", "R-L", "CIDEr", "F_BERT", "Cat"])
    ap.add_argument("--attr", action="store_true", help="print per-attribute CIDEr rows (Table 3)")
    ap.add_argument("--out")
    a = ap.parse_args()

    summary = {}
    for cfg in a.configs:
        files = sorted(glob.glob(f"{a.results}/{cfg}/seed*.json"))
        if not files:
            print(f"[warn] no results for {cfg}")
            continue
        runs = [load_json(f) for f in files]
        row = {}
        for m in a.metrics:
            vals = [r["corpus"][m] for r in runs if m in r["corpus"]]
            if vals:
                row[m] = {"mean": statistics.mean(vals),
                          "std": statistics.stdev(vals) if len(vals) > 1 else None,
                          "n_seeds": len(vals), "values": vals}
        if a.attr:
            row["per_attribute"] = {}
            for at in ATTRS:
                vals = [r["per_attribute"][at]["CIDEr"] for r in runs if at in r.get("per_attribute", {})]
                if vals:
                    row["per_attribute"][at] = {"mean": statistics.mean(vals),
                                                "std": statistics.stdev(vals) if len(vals) > 1 else None}
        summary[cfg] = row

    print("\n=== mean ± std over seeds ===")
    for cfg, row in summary.items():
        n = max((v["n_seeds"] for k, v in row.items() if k != "per_attribute"), default=0)
        cells = [f"{m}={row[m]['mean']:.2f}" + (f"±{row[m]['std']:.2f}" if row[m]["std"] is not None else "")
                 for m in a.metrics if m in row]
        print(f"{cfg:>12s} ({n} seeds): " + "  ".join(cells))

    print("\n=== LaTeX cells (paste into Table 1 / Table 2a) ===")
    for cfg, row in summary.items():
        print(f"% {cfg}\n" + " & ".join(fmt(row[m]["mean"], row[m]["std"]) if m in row else "--"
                                         for m in a.metrics) + " \\\\")

    if a.attr:
        print("\n=== LaTeX per-attribute CIDEr (Table 3 column order) ===")
        for cfg, row in summary.items():
            pa = row.get("per_attribute", {})
            print(f"% {cfg}\n" + " & ".join(f"{pa[at]['mean']:.2f}" if at in pa else "--" for at in ATTRS) + " \\\\")

    if a.out:
        save_json(summary, a.out)
        print(f"\n[ok] saved {a.out}")

if __name__ == "__main__":
    main()
