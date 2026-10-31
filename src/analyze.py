"""Analysis of results/scores.csv -> results/analysis/*.csv, results/analysis/summary.json, figures/*.png.

All CIs are 95% cluster bootstrap over the 40 user tasks (B=2000, seed 42). Paired tests use a task-clustered
sign-flip permutation test of the mean paired difference (10k permutations). Holm correction within each hypothesis
family (across models).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS
from instructions import IDS, GROUP

OUT, FIG = ROOT / "results/analysis", ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
B, SEED = 2000, 42
ORDER = sorted([m for m in MODELS if not MODELS[m].get("variant")], key=lambda m: (MODELS[m]["params"], m))


def task_matrix(df, val):
    """Per-task sums and counts (tasks are the bootstrap clusters)."""
    g = df.groupby("task_id")[val].agg(["sum", "count"]).reindex(range(40), fill_value=0)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def boot_mean(df, val="ok"):
    s, n = task_matrix(df, val)
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(40, np.full(40, 1 / 40), size=B)
    est = s.sum() / n.sum()
    bs = (w @ s) / np.maximum(w @ n, 1)
    return est, *np.percentile(bs, [2.5, 97.5])


def perm_test(df, val="d"):
    """Sign-flip permutation test on task-level sums of paired differences (H0: mean diff = 0)."""
    s, n = task_matrix(df, val)
    obs = s.sum() / n.sum()
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1.0, 1.0], size=(10000, 40))
    null = (signs @ s) / n.sum()
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj


def fmt(e, lo, hi, pct=True):
    f = 100 if pct else 1
    return f"{e * f:.1f} [{lo * f:.1f}, {hi * f:.1f}]"


def main():
    full = pd.read_csv(ROOT / "results/scores.csv")
    df = full[~full.model.str.contains("#")]
    models = [m for m in ORDER if m in df.model.unique()]
    summary = {"models": models, "n_rows": len(df)}

    # ---------------- E1: load curve
    e1 = df[df.exp == "E1"]
    rows = []
    for m in models:
        for k in sorted(e1.k.unique()):
            d = e1[(e1.model == m) & (e1.k == k)]
            e, lo, hi = boot_mean(d)
            allok = d.groupby("pid").agg(ok=("ok", "min"), task_id=("task_id", "first"))
            ea, la, ha = boot_mean(allok)
            rows.append(dict(model=m, k=k, compliance=e, lo=lo, hi=hi, all_ok=ea, all_lo=la, all_hi=ha, n=len(d)))
    curve = pd.DataFrame(rows); curve.to_csv(OUT / "e1_curve.csv", index=False)
    # decay slope: linear-probability slope of per-instruction compliance on k, with instruction fixed effects
    slopes = []
    for m in models:
        d = e1[e1.model == m].copy()
        d["okc"] = d.ok - d.groupby("iid").ok.transform("mean")      # within-instruction demeaning
        d["kc"] = d.k - d.groupby("iid").k.transform("mean")
        def slope(dd):
            return (dd.okc * dd.kc).sum() / (dd.kc ** 2).sum()
        rng = np.random.default_rng(SEED)
        groups = [g for _, g in d.groupby("task_id")]
        bs = []
        for _ in range(500):
            idx = rng.integers(0, len(groups), len(groups))
            bs.append(slope(pd.concat([groups[i] for i in idx])))
        c1 = curve[(curve.model == m) & (curve.k == 1)].compliance.item()
        c5p = e1[(e1.model == m) & (e1.k >= 5)].ok.mean()
        slopes.append(dict(model=m, params=MODELS[m]["params"], family=MODELS[m]["family"], slope_pp_per_instr=100 * slope(d),
                           lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                           k1=c1, k5plus=c5p, rel_drop=(c1 - c5p) / c1))
    slopes = pd.DataFrame(slopes)
    # logit-scale slope (robust to floor/ceiling compression): GLM ok ~ k + C(iid), cluster-robust SE by task
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    for i, m in enumerate(slopes.model):
        d = e1[e1.model == m].copy(); d["ok"] = d.ok.astype(int)
        fit = smf.glm("ok ~ k + C(iid)", data=d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d.task_id})
        b, se = fit.params["k"], fit.bse["k"]
        slopes.loc[i, "logit_slope"] = b; slopes.loc[i, "logit_lo"] = b - 1.96 * se; slopes.loc[i, "logit_hi"] = b + 1.96 * se
        slopes.loc[i, "odds_ratio_per_instr"] = np.exp(b)
    slopes.to_csv(OUT / "e1_slopes.csv", index=False)
    rho3, p3 = spearmanr(np.log(slopes.params), slopes.logit_slope)
    rho, p = spearmanr(np.log(slopes.params), slopes.slope_pp_per_instr)
    rho2, p2 = spearmanr(np.log(slopes.params), slopes.rel_drop)
    summary["H5_spearman_logparams_vs_slope"] = dict(rho=rho, p=p, n=len(slopes))
    summary["H5_spearman_logparams_vs_reldrop"] = dict(rho=rho2, p=p2, n=len(slopes))
    summary["H5_spearman_logparams_vs_logit_slope"] = dict(rho=rho3, p=p3, n=len(slopes))
    per_instr = e1.groupby(["model", "k", "iid"]).ok.mean().unstack("iid")
    per_instr.to_csv(OUT / "e1_per_instruction.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(models):
        c = curve[curve.model == m]
        col = cmap(i / max(1, len(models) - 1))
        ls = "--" if MODELS[m]["family"] == "llama3" else "-"
        axes[0].errorbar(c.k, 100 * c.compliance, yerr=[100 * (c.compliance - c.lo), 100 * (c.hi - c.compliance)],
                         marker="o", ls=ls, color=col, label=m, capsize=3)
        axes[1].errorbar(c.k, 100 * c.all_ok, yerr=[100 * (c.all_ok - c.all_lo), 100 * (c.all_hi - c.all_ok)],
                         marker="o", ls=ls, color=col, label=m, capsize=3)
    for ax, t in zip(axes, ["Per-instruction compliance", "All instructions satisfied"]):
        ax.axhspan(50, 65, color="grey", alpha=0.12)
        ax.set_xlabel("number of instructions in system prompt (k)"); ax.set_ylabel("% of responses"); ax.set_title(t)
        ax.set_xticks([1, 2, 3, 5, 7, 10]); ax.set_ylim(0, 102); ax.grid(alpha=0.3)
    axes[0].axhline(90, color="grey", ls=":", lw=1)
    axes[1].legend(fontsize=8)
    fig.suptitle("E1: compliance vs instruction load (shaded: hypothesised 50-65% band at k>=5)")
    fig.tight_layout(); fig.savefig(FIG / "e1_load_curve.png", dpi=150); plt.close(fig)

    # ---------------- E2: pairwise interference, paired with the same (instruction, task) alone (E1 k=1)
    alone = df[(df.exp == "E1") & (df.k == 1)].set_index(["model", "iid", "task_id"]).ok
    e2 = df[df.exp == "E2"].copy()
    e2["partner"] = e2.others
    e2["base"] = alone.reindex(pd.MultiIndex.from_frame(e2[["model", "iid", "task_id"]])).to_numpy()
    e2["d"] = e2.ok - e2.base
    e2["pgroup"] = e2.partner.map(GROUP)
    rows, ps = [], []
    for m in models:
        for tg in ("PS", "FL"):
            for pg in ("FL", "PS"):
                d = e2[(e2.model == m) & (e2.group == tg) & (e2.pgroup == pg)]
                e, lo, hi = boot_mean(d, "d")
                rows.append(dict(model=m, target_group=tg, partner_group=pg, alone=d.base.mean(), paired=d.ok.mean(),
                                 delta=e, lo=lo, hi=hi, p=perm_test(d), n=len(d)))
        # H2 contrast: PS targets, FL partner minus PS partner (difference of deltas), task-clustered bootstrap
    g2 = pd.DataFrame(rows)
    g2["p_holm"] = holm(g2.p)
    g2.to_csv(OUT / "e2_group_deltas.csv", index=False)
    # contrast for H2 per model: delta(PS|FL) - delta(PS|PS)
    con = []
    for m in models:
        d = e2[(e2.model == m) & (e2.group == "PS")].copy()
        d["c"] = np.where(d.pgroup == "FL", d.d, 0.0)
        d2 = d.copy(); d2["c"] = np.where(d.pgroup == "PS", d.d, 0.0)
        s1, n1 = task_matrix(d[d.pgroup == "FL"], "d"); s2, n2 = task_matrix(d[d.pgroup == "PS"], "d")
        rng = np.random.default_rng(SEED); w = rng.multinomial(40, np.full(40, 1 / 40), size=B)
        bs = (w @ s1) / np.maximum(w @ n1, 1) - (w @ s2) / np.maximum(w @ n2, 1)
        est = s1.sum() / n1.sum() - s2.sum() / n2.sum()
        con.append(dict(model=m, delta_PS_given_FL=s1.sum() / n1.sum(), delta_PS_given_PS=s2.sum() / n2.sum(),
                        contrast=est, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    pd.DataFrame(con).to_csv(OUT / "e2_h2_contrast.csv", index=False)
    # 10x10 matrices: pooled over models and per model
    mat = e2.groupby(["iid", "partner"]).d.mean().unstack().reindex(index=IDS, columns=IDS)
    mat.to_csv(OUT / "e2_matrix_pooled.csv")
    for m in models:
        e2[e2.model == m].groupby(["iid", "partner"]).d.mean().unstack().reindex(index=IDS, columns=IDS) \
            .to_csv(OUT / f"e2_matrix_{m}.csv")
    # matrix restricted to "capable" models (single-instruction compliance >= 80%) to avoid floor-level noise
    k1 = curve[curve.k == 1].set_index("model").compliance
    capable = [m for m in models if k1[m] >= 0.8]
    summary["capable_models"] = capable
    mat_c = e2[e2.model.isin(capable)].groupby(["iid", "partner"]).d.mean().unstack().reindex(index=IDS, columns=IDS)
    mat_c.to_csv(OUT / "e2_matrix_capable.csv")
    for name, M, ttl in (("e2_interference_matrix.png", mat, "pooled over all models"),
                         ("e2_interference_matrix_capable.png", mat_c, "models with single-instr. compliance >= 80%")):
        draw_matrix(M, ttl, FIG / name)
    main_rest(df, full, models, curve, e1, summary, slopes)


def draw_matrix(mat, ttl, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    v = 42
    im = ax.imshow(100 * mat.to_numpy(), cmap="RdBu", vmin=-v, vmax=v)
    ax.set_xticks(range(10)); ax.set_xticklabels(IDS, rotation=60, ha="right"); ax.set_yticks(range(10)); ax.set_yticklabels(IDS)
    for i in range(10):
        for j in range(10):
            if not np.isnan(mat.iat[i, j]):
                ax.text(j, i, f"{100 * mat.iat[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.axhline(3.5, color="k", lw=1); ax.axvline(3.5, color="k", lw=1)
    ax.set_xlabel("partner instruction (B)"); ax.set_ylabel("target instruction (A)")
    ax.set_title(f"E2: change in A's compliance when paired with B\n(pp vs A alone; {ttl})")
    fig.colorbar(im, ax=ax, label="pp"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main_rest(df, full, models, curve, e1, summary, slopes):

    # ---------------- E3: first vs last (paired on target, task)
    e3 = df[(df.exp == "E3") & df.is_target]
    piv = e3.pivot_table(index=["model", "target", "task_id"], columns="cond", values="ok").reset_index()
    piv["d"] = piv["last"] - piv["first"]
    rows = []
    for m in models:
        d = piv[piv.model == m]
        e, lo, hi = boot_mean(d, "d")
        rows.append(dict(model=m, first=d["first"].mean(), last=d["last"].mean(), delta=e, lo=lo, hi=hi,
                         p=perm_test(d), n_pairs=len(d), n_discordant=int((d.d != 0).sum())))
    d = piv; e, lo, hi = boot_mean(d, "d")
    rows.append(dict(model="ALL", first=d["first"].mean(), last=d["last"].mean(), delta=e, lo=lo, hi=hi,
                     p=perm_test(d), n_pairs=len(d), n_discordant=int((d.d != 0).sum())))
    e3r = pd.DataFrame(rows); e3r["p_holm"] = list(holm(e3r.p[:-1])) + [e3r.p.iloc[-1]]
    e3r.to_csv(OUT / "e3_position.csv", index=False)
    piv.groupby(["model", "target"]).d.mean().unstack().to_csv(OUT / "e3_position_by_instruction.csv")
    # also: non-target instructions in E3 - does putting the target first/last shift the others?
    # secondary: position effect inside E1 (random orders, k>=5): compliance by normalised position
    e1p = e1[e1.k >= 5].copy(); e1p["relpos"] = e1p.pos / (e1p.n_lines - 1)
    e1p["third"] = pd.cut(e1p.relpos, [-0.01, 0.34, 0.67, 1.0], labels=["first third", "middle", "last third"])
    e1p.groupby(["model", "third"], observed=True).ok.mean().unstack().to_csv(OUT / "e1_position_thirds.csv")

    # ---------------- E4: repetition
    e4 = df[df.exp == "E4"]
    tgt = e4[e4.is_target].pivot_table(index=["model", "target", "task_id"], columns="cond", values="ok").reset_index()
    oth = e4[~e4.is_target].groupby(["model", "target", "task_id", "cond"]).ok.mean().unstack().reset_index()
    rows = []
    for which, tab in (("target", tgt), ("others", oth)):
        for m in models + ["ALL"]:
            d = tab if m == "ALL" else tab[tab.model == m]
            r = dict(model=m, which=which, base=d["base"].mean(), repeat=d["repeat"].mean(), filler=d["filler"].mean())
            for a, b in (("repeat", "base"), ("repeat", "filler"), ("filler", "base")):
                dd = d.assign(d=d[a] - d[b])
                e, lo, hi = boot_mean(dd, "d")
                r[f"{a}-{b}"] = e; r[f"{a}-{b}_lo"] = lo; r[f"{a}-{b}_hi"] = hi; r[f"{a}-{b}_p"] = perm_test(dd)
            rows.append(r)
    e4r = pd.DataFrame(rows)
    for which in ("target", "others"):
        mask = (e4r.which == which) & (e4r.model != "ALL")
        for c in ("repeat-base", "repeat-filler"):
            e4r.loc[mask, f"{c}_p_holm"] = holm(e4r.loc[mask, f"{c}_p"])
    e4r.to_csv(OUT / "e4_repetition.csv", index=False)
    tgt.assign(d=tgt["repeat"] - tgt["base"]).groupby(["model", "target"]).d.mean().unstack().to_csv(OUT / "e4_repeat_by_instruction.csv")

    # ---------------- Figures for E3/E4
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(len(models))
    r3 = e3r[e3r.model != "ALL"]
    axes[0].bar(x - 0.2, 100 * r3["first"], 0.4, label="target first"); axes[0].bar(x + 0.2, 100 * r3["last"], 0.4, label="target last")
    axes[0].set_title("E3: target compliance by position (k=5)"); axes[0].legend()
    for ax_i, which in ((1, "target"), (2, "others")):
        r4 = e4r[(e4r.which == which) & (e4r.model != "ALL")]
        for j, c in enumerate(("base", "filler", "repeat")):
            axes[ax_i].bar(x + (j - 1) * 0.27, 100 * r4[c], 0.27, label=c)
        axes[ax_i].set_title(f"E4: {'repeated instruction' if which == 'target' else 'other 4 instructions'}")
        axes[ax_i].legend()
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right"); ax.set_ylabel("% compliant"); ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "e3_e4_position_repetition.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for _, r in slopes.iterrows():
        ax.errorbar(r.params, r.slope_pp_per_instr, yerr=[[r.slope_pp_per_instr - r.lo], [r.hi - r.slope_pp_per_instr]],
                    marker="s" if r.family == "llama3" else "o", capsize=3)
        ax.annotate(r.model, (r.params, r.slope_pp_per_instr), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("parameters (B, log scale)"); ax.set_ylabel("pp change per added instruction")
    ax.set_title("H5: compliance decay slope vs model size"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "h5_slope_vs_size.png", dpi=150); plt.close(fig)

    # ---------------- Robustness: numbered-list rendering of the same E1 (k=1,5,10) and E3 prompts
    var = full[full.model.str.contains("#")]
    rob = []
    for vm in sorted(var.model.unique()):
        base_m = vm.split("#")[0]
        for label, dd in (("plain", df[df.model == base_m]), ("numbered", var[var.model == vm])):
            r = dict(model=base_m, format=label)
            for k in (1, 5, 10):
                r[f"E1_k{k}"] = dd[(dd.exp == "E1") & (dd.k == k)].ok.mean()
            t3 = dd[(dd.exp == "E3") & dd.is_target].pivot_table(index=["target", "task_id"], columns="cond", values="ok")
            r["E3_first"], r["E3_last"] = t3["first"].mean(), t3["last"].mean()
            dd3 = t3.reset_index().assign(d=lambda x: x["last"] - x["first"])
            e, lo, hi = boot_mean(dd3, "d"); r["E3_last_minus_first"], r["lo"], r["hi"] = e, lo, hi
            r["E3_p"] = perm_test(dd3)
            rob.append(r)
    if rob:
        rob = pd.DataFrame(rob); rob.to_csv(OUT / "robustness_numbered.csv", index=False); print(rob.round(3).to_string())

    summary["truncation_rate"] = df.drop_duplicates("pid model".split()).groupby("model").truncated.mean().to_dict()
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(curve.round(3).to_string()); print(slopes.round(3).to_string()); print(summary)
    print(pd.read_csv(OUT / "e2_group_deltas.csv").round(3).to_string()); print(pd.read_csv(OUT / "e2_h2_contrast.csv").round(3).to_string())
    print(e3r.round(3).to_string()); print(e4r.round(3).to_string())


if __name__ == "__main__":
    main()
