#!/usr/bin/env python3
"""Merge partial XRL comparison results into the final outputs.

Run 9 (full, run9 checkpoint) was killed by a VM restart during DQN
ate, but its partial_results.json already holds: heuristic (ate + hte
+ fidelity, complete) and DQN hte. A second, DQN-only run then
produced DQN hte + ate + fidelity (hte is re-run because fidelity
needs the in-memory hte scm). This script merges the two partials
into the final comparison_table.csv/.md + per_run_details.json in the
main output dir, recomputes the trained-vs-heuristic graph comparison
from the merged edge sets, and records the provenance of every number
(which run it came from). No number is invented: every value comes
from one of the two partial JSONs, both produced by real BART runs
(see their logs). The killed run's own DQN hte is superseded by the
DQN-only rerun's hte in the merged table, but is recorded in
provenance as a cross-check (same checkpoint, same seed).

Usage:
    .venv-ablation/bin/python -B scripts/merge_xrl_comparison.py \
        --base results/xrl-trained-vs-heuristic/partial_results.json \
        --extra results/xrl-trained-vs-heuristic-dqn-only/partial_results.json \
        --out-dir results/xrl-trained-vs-heuristic
"""
import argparse
import csv
import json
import shutil
from pathlib import Path


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True,
                    help="partial_results.json with heuristic (complete) "
                    "+ DQN hte (killed run; kept as cross-check)")
    ap.add_argument("--extra", type=Path, required=True,
                    help="partial_results.json with DQN hte + ate + "
                    "fidelity (DQN-only rerun)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    base = json.loads(args.base.read_text())
    extra = json.loads(args.extra.read_text())
    out_dir = args.out_dir
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # ---- merge rows: base rows first, extra rows replace same policy+method
    rows = {(r["policy"], r["method"]): dict(r) for r in base["rows"]}
    for r in extra["rows"]:
        rows[(r["policy"], r["method"])] = dict(r)
    order = [("heuristic", "hte"), ("heuristic", "ate"),
             ("dqn", "hte"), ("dqn", "ate")]
    merged_rows = [rows[k] for k in order if k in rows]

    # ---- merge details
    details = json.loads(json.dumps(base["details"]))
    details["provenance"] = {
        "heuristic": "run9 full run (2026-11-11 11:05-11:09, killed by VM "
                     "restart during DQN ate; comparison_run.log in this "
                     "dir; partial_results.json holds heuristic complete)",
        "dqn_hte_killed_run_crosscheck": "run9 full run, DQN hte 14 edges, "
                     "infer 1738.58 s, push parents [push, x, v, theta, w] "
                     "— SUPERSEDED in the merged table by the DQN-only "
                     "rerun's hte (same checkpoint, same seed); kept here "
                     "as a cross-check, not used for any merged number",
        "dqn_hte_ate_fidelity": "DQN-only rerun (same run9 checkpoint, "
                     "seed 0, 200 episodes), see "
                     "results/xrl-trained-vs-heuristic-dqn-only-run9/",
        "checkpoint": base["details"].get("checkpoint"),
        "note": "Merged DQN hte/ate/fidelity all come from the DQN-only "
                "rerun on the run9 checkpoint (500.00/500 on 100-ep eval); "
                "heuristic rows come from the killed full run on the same "
                "checkpoint. The old run5-checkpoint comparison is "
                "archived separately in "
                "results/xrl-trained-vs-heuristic-run5/.",
    }
    for pname, pruns in extra["details"]["runs"].items():
        if pname not in details["runs"]:
            details["runs"][pname] = pruns
            continue
        details["runs"][pname].setdefault("methods", {}).update(
            pruns.get("methods", {}))
        if "fidelity_eval_notebook" in pruns:
            details["runs"][pname]["fidelity_eval_notebook"] = \
                pruns["fidelity_eval_notebook"]
        for k in ("n_transitions", "rollout_s", "mean_ep_len",
                  "action1_frac"):
            if k in pruns:
                details["runs"][pname][k] = pruns[k]

    # ---- recompute trained-vs-heuristic comparison from merged edges
    edges = {}
    for r in merged_rows:
        edges[(r["policy"], r["method"])] = set(
            e.strip() for e in r["edges"].split(";") if e.strip())
    comparison = {}
    for method in ("hte", "ate"):
        h = edges.get(("heuristic", method), set())
        d = edges.get(("dqn", method), set())
        comparison[method] = {
            "heuristic_n_edges": len(h), "dqn_n_edges": len(d),
            "jaccard": round(jaccard(h, d), 4),
            "symdiff": len(h ^ d),
            "only_heuristic": sorted(h - d),
            "only_dqn": sorted(d - h),
            "common": sorted(h & d),
        }
    details["comparison_trained_vs_heuristic"] = comparison

    # ---- copy the DQN-only graphs into the main graphs dir
    extra_graphs = args.extra.parent / "graphs"
    if extra_graphs.is_dir():
        for png in extra_graphs.glob("graph_dqn_*.png"):
            shutil.copy2(png, graphs_dir / png.name)
            print(f"copied {png.name}")

    # ---- write final outputs (same format as run_xrl_comparison.py)
    label = details.get("label", "")
    ckpt_eval = details.get("checkpoint_final_eval_100ep", {})
    fieldnames = ["policy", "method", "n_transitions", "n_edges", "edges",
                  "edges_collapsed", "per_target_causes", "time_s",
                  "rollout_s", "fid_valid_steps", "fid_n_test_dt_lr",
                  "dt_acc", "dt_f1", "lr_acc", "lr_f1", "bart_acc",
                  "bart_f1", "bart_n_test", "bart_test_capped",
                  "global_cause_push_parents"]
    with open(out_dir / "comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in merged_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    (out_dir / "per_run_details.json").write_text(
        json.dumps(details, indent=2, default=str))

    md = []
    md.append("# CausalXRL extension D — XRL trained-vs-heuristic "
              "comparison (CartPole)\n")
    md.append(f"> **{label}**\n")
    md.append(f"> Checkpoint DQN: `{details.get('checkpoint')}` "
              f"(final eval 100ep from checkpoint metadata: {ckpt_eval}). "
              f"Episodes: {details.get('episodes')} per policy, seed "
              f"{details.get('seed')}. "
              "Data: gymnasium CartPole-v1 rollouts (repo `play()` uses gym "
              "0.20, incompatible with Python 3.12, so gymnasium is used "
              "for data collection only; the repo's transition-recording "
              "rule is kept). `hte` = repo `infer_causes_hte` (70/30 "
              "split); `ate` = repo `infer_causes_ate` (50/50 split). "
              "Fidelity is eval-notebook style: steps whose local causal "
              "graph has >1 cause for `push`, 60/40 split "
              "(random_state=10); DT/LR on the full held-out set, "
              "BART-weighted on at most 200 test steps (see per-run "
              "details). Global-cause set for BART-weighted = learned "
              "parents of `push` in the hte graph (not the notebook's "
              "hard-coded list). Time = structure-learning seconds "
              "(infer only, excluding rollout and fidelity). "
              "Provenance: heuristic from the run9 full run (killed by a "
              "VM restart during DQN ate); DQN hte/ate/fidelity from a "
              "DQN-only rerun on the same run9 checkpoint (see "
              "`per_run_details.json` > provenance; the killed run's own "
              "DQN hte is recorded there as a cross-check). Numbers "
              "vary ±1–2 edges between repeated BART runs — see the "
              "reproducibility note in the report.\n")
    md.append("| policy | method | n_trans | n_edges | time_s (infer) | "
              "DT acc / F1 | LR acc / F1 | BART acc / F1 (n_test) | edges |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in merged_rows:
        dt = (f"{r.get('dt_acc')} / {r.get('dt_f1')}"
              if r.get("dt_acc") is not None else "—")
        lr = (f"{r.get('lr_acc')} / {r.get('lr_f1')}"
              if r.get("lr_acc") is not None else "—")
        bart = (f"{r.get('bart_acc')} / {r.get('bart_f1')} "
                f"(n={r.get('bart_n_test')})"
                if r.get("bart_acc") is not None else "—")
        md.append(f"| {r['policy']} | {r['method']} | {r['n_transitions']} "
                  f"| {r['n_edges']} | {r['time_s']} | {dt} | {lr} | {bart} "
                  f"| {r['edges']} |")
    md.append("")
    md.append("## Trained (DQN) vs heuristic — graph differences\n")
    for method, comp in comparison.items():
        md.append(f"### {method}\n")
        md.append(f"- Jaccard = {comp['jaccard']}, symmetric difference = "
                  f"{comp['symdiff']} edges "
                  f"(heuristic {comp['heuristic_n_edges']} edges, DQN "
                  f"{comp['dqn_n_edges']} edges).")
        md.append(f"- Only in heuristic: "
                  f"`{'; '.join(comp['only_heuristic']) or '—'}`")
        md.append(f"- Only in DQN: `{'; '.join(comp['only_dqn']) or '—'}`")
        md.append(f"- Common: `{'; '.join(comp['common']) or '—'}`")
        md.append("")
    md.append("## Per-target causes\n")
    for r in merged_rows:
        md.append(f"- **{r['policy']} / {r['method']}**: "
                  f"`{r['per_target_causes']}`")
    (out_dir / "comparison_table.md").write_text("\n".join(md) + "\n")
    print(f"merged {len(merged_rows)} rows -> {out_dir}")
    for method, comp in comparison.items():
        print(f"  {method}: Jaccard={comp['jaccard']} "
              f"symdiff={comp['symdiff']} "
              f"(heur {comp['heuristic_n_edges']} / "
              f"dqn {comp['dqn_n_edges']} edges)")


if __name__ == "__main__":
    main()
