from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = RESULTS / "plots"
OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# utils
# -----------------------------
def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_read_csv(path: Path):
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[skip] failed to read {path}: {e}")
        return None


def _method_order(methods):
    # 원하는 발표 순서 우선
    preferred = [
        "Reactive_SlotBest",
        "Predicted_Diffusion",
        "Predicted_Consistency",
        "Learned_TC_Offline",
        "Learned_TC_RTCorrected",
        "Sustainable_DAG_NetworkX",
    ]
    rank = {m: i for i, m in enumerate(preferred)}
    return sorted(methods, key=lambda x: rank.get(x, 999))


def _sort_df_by_method(df, method_col="Method"):
    methods = list(df[method_col].dropna().unique())
    order = _method_order(methods)
    df["__order"] = df[method_col].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values(["__order"]).drop(columns="__order")
    return df


def _save_bar_plot(df, x_col, y_col, title, y_label, out_name, rotate=25):
    if y_col is None or y_col not in df.columns:
        print(f"[skip] {title} - y_col missing")
        return
    if x_col not in df.columns:
        print(f"[skip] {title} - x_col missing")
        return

    plt.figure(figsize=(9, 4.8))
    plt.bar(df[x_col], df[y_col])
    plt.xticks(rotation=rotate, ha="right")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=150)
    plt.close()


# -----------------------------
# Exp1
# -----------------------------
def plot_exp1_core_compare():
    f = RESULTS / "exp1_core_compare_summary.csv"
    df = _safe_read_csv(f)
    if df is None:
        return

    qoe_col = _pick_col(df, ["EffectiveQoE_mean", "QoE_mean"])
    dl_col  = _pick_col(df, ["DeadlineMissRatio_mean", "DLMiss_mean"])
    av_col  = _pick_col(df, ["Availability_mean", "Avail_mean"])
    lat_col = _pick_col(df, ["MeanLatency_ms_mean", "Lat_ms_mean"])
    int_col = _pick_col(df, ["MeanInterruption_ms_mean", "Int_ms_mean"])

    if "Method" in df.columns:
        df = _sort_df_by_method(df, "Method")

    metrics = [
        ("Availability", av_col),
        ("EffectiveQoE", qoe_col),
        ("DeadlineMissRatio", dl_col),
        ("MeanLatency_ms", lat_col),
        ("MeanInterruption_ms", int_col),
    ]

    for metric_name, col in metrics:
        if col is None:
            continue
        d = df[["Method", "SamplingSteps", col]].copy()
        # 같은 Method가 여러 줄인 경우 방어 (평균)
        d = d.groupby("Method", as_index=False)[col].mean()
        d = _sort_df_by_method(d, "Method")

        _save_bar_plot(
            d, "Method", col,
            title=f"Exp1 Core Compare - {metric_name}",
            y_label=metric_name,
            out_name=f"exp1_{metric_name}.png",
            rotate=20
        )

    print("[ok] exp1 plots saved")


# -----------------------------
# Exp2
# -----------------------------
def plot_exp2_step_sweep():
    f = RESULTS / "exp2_step_sweep_summary.csv"
    df = _safe_read_csv(f)
    if df is None:
        return

    qoe_col = _pick_col(df, ["EffectiveQoE_mean", "QoE_mean"])
    dl_col  = _pick_col(df, ["DeadlineMissRatio_mean", "DLMiss_mean"])
    av_col  = _pick_col(df, ["Availability_mean", "Avail_mean"])

    if "Method" not in df.columns or "SamplingSteps" not in df.columns:
        print("[skip] exp2 summary missing Method/SamplingSteps")
        return

    for metric_name, col in [
        ("DeadlineMissRatio", dl_col),
        ("EffectiveQoE", qoe_col),
        ("Availability", av_col),
    ]:
        if col is None:
            continue

        plt.figure(figsize=(8, 4.8))
        # method 순서 고정
        method_order = _method_order(df["Method"].dropna().unique())
        for method in method_order:
            g = df[df["Method"] == method].copy()
            if g.empty:
                continue
            g = g.sort_values("SamplingSteps")
            plt.plot(g["SamplingSteps"], g[col], marker="o", label=method)

        plt.xlabel("Sampling Steps")
        plt.ylabel(metric_name)
        plt.title(f"Exp2 Step Sweep - {metric_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"exp2_step_sweep_{metric_name}.png", dpi=150)
        plt.close()

    print("[ok] exp2 plots saved")


# -----------------------------
# Exp3 (algorithm compare from experiments_atg_leo.py)
# -----------------------------
def plot_exp3_algorithm_compare():
    f = RESULTS / "exp3_algorithm_compare_summary.csv"
    df = _safe_read_csv(f)
    if df is None:
        return

    qoe_col = _pick_col(df, ["EffectiveQoE_mean", "QoE_mean"])
    dl_col  = _pick_col(df, ["DeadlineMissRatio_mean", "DLMiss_mean"])
    av_col  = _pick_col(df, ["Availability_mean", "Avail_mean"])
    lat_col = _pick_col(df, ["MeanLatency_ms_mean", "Lat_ms_mean"])
    int_col = _pick_col(df, ["MeanInterruption_ms_mean", "Int_ms_mean"])

    if "Method" not in df.columns:
        print("[skip] exp3 algorithm summary missing Method")
        return

    # SamplingSteps 중복 방어용 평균
    for metric_name, col in [
        ("Availability", av_col),
        ("EffectiveQoE", qoe_col),
        ("DeadlineMissRatio", dl_col),
        ("MeanLatency_ms", lat_col),
        ("MeanInterruption_ms", int_col),
    ]:
        if col is None:
            continue
        d = df[["Method", col]].copy().groupby("Method", as_index=False)[col].mean()
        d = _sort_df_by_method(d, "Method")

        _save_bar_plot(
            d, "Method", col,
            title=f"Exp3 Algorithm Compare - {metric_name}",
            y_label=metric_name,
            out_name=f"exp3_algorithm_compare_{metric_name}.png",
            rotate=25
        )

    print("[ok] exp3 algorithm compare plots saved")


# -----------------------------
# Exp3 (learned realtime from train_eval_realtime_proposed_atg_leo.py)
# -----------------------------
def plot_exp3_learned_realtime():
    f = RESULTS / "exp3_learned_realtime_summary.csv"
    df = _safe_read_csv(f)
    if df is None:
        return

    qoe_col = _pick_col(df, ["EffectiveQoE_mean", "QoE_mean"])
    dl_col  = _pick_col(df, ["DeadlineMissRatio_mean", "DLMiss_mean"])
    av_col  = _pick_col(df, ["Availability_mean", "Avail_mean"])
    lat_col = _pick_col(df, ["MeanLatency_ms_mean", "Lat_ms_mean"])

    if "Method" not in df.columns:
        print("[skip] exp3 learned realtime summary missing Method")
        return

    for metric_name, col in [
        ("Availability", av_col),
        ("EffectiveQoE", qoe_col),
        ("DeadlineMissRatio", dl_col),
        ("MeanLatency_ms", lat_col),
    ]:
        if col is None:
            continue
        d = df[["Method", col]].copy()
        d = d.groupby("Method", as_index=False)[col].mean()
        d = _sort_df_by_method(d, "Method")

        _save_bar_plot(
            d, "Method", col,
            title=f"Exp3 Learned/Realtime Compare - {metric_name}",
            y_label=metric_name,
            out_name=f"exp3_learned_realtime_{metric_name}.png",
            rotate=25
        )

    print("[ok] exp3 learned realtime plots saved")


# -----------------------------
# Train history
# -----------------------------
def plot_train_history():
    f = RESULTS / "train_history_real_env.csv"
    df = _safe_read_csv(f)
    if df is None:
        return

    if "epoch" not in df.columns:
        print("[skip] train history missing epoch")
        return

    plot_targets = [
        ("avg_total_loss", "Train History - Avg Total Loss", "Loss", "train_avg_total_loss.png"),
        ("avg_tf_loss", "Train History - Avg Transformer Loss", "Loss", "train_avg_tf_loss.png"),
        ("avg_rl_loss", "Train History - Avg RL Surrogate Loss", "Loss", "train_avg_rl_loss.png"),
        ("avg_cs_loss", "Train History - Avg Consistency Loss", "Loss", "train_avg_cs_loss.png"),
        ("avg_reward", "Train History - Avg Reward", "Reward", "train_avg_reward.png"),
    ]

    for col, title, ylab, out_name in plot_targets:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 4.5))
        plt.plot(df["epoch"], df[col], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylab)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / out_name, dpi=150)
        plt.close()

    print("[ok] train history plots saved")


# -----------------------------
# Unified comparison (optional)
# -----------------------------
def plot_exp3_unified_if_possible():
    """
    exp3_algorithm_compare_summary + exp3_learned_realtime_summary 둘 다 있으면 합쳐서 한 장씩 그림 생성.
    """
    f_algo = RESULTS / "exp3_algorithm_compare_summary.csv"
    f_lr   = RESULTS / "exp3_learned_realtime_summary.csv"
    df_algo = _safe_read_csv(f_algo)
    df_lr = _safe_read_csv(f_lr)

    if df_algo is None or df_lr is None:
        print("[skip] unified exp3 plot requires both algorithm_compare and learned_realtime summaries")
        return

    if "Method" not in df_algo.columns or "Method" not in df_lr.columns:
        print("[skip] unified exp3 summary missing Method")
        return

    # 필요한 컬럼만 맞춰서 concat
    # *_mean 컬럼 기준으로 통일
    target_metric_map = {
        "Availability": ["Availability_mean", "Avail_mean"],
        "EffectiveQoE": ["EffectiveQoE_mean", "QoE_mean"],
        "DeadlineMissRatio": ["DeadlineMissRatio_mean", "DLMiss_mean"],
    }

    for metric_name, candidates in target_metric_map.items():
        col_algo = _pick_col(df_algo, candidates)
        col_lr = _pick_col(df_lr, candidates)
        if col_algo is None and col_lr is None:
            continue

        rows = []
        if col_algo is not None:
            tmp = df_algo[["Method", col_algo]].copy().rename(columns={col_algo: "value"})
            rows.append(tmp)
        if col_lr is not None:
            tmp = df_lr[["Method", col_lr]].copy().rename(columns={col_lr: "value"})
            rows.append(tmp)

        if not rows:
            continue

        d = pd.concat(rows, axis=0, ignore_index=True)
        d = d.groupby("Method", as_index=False)["value"].mean()
        d = _sort_df_by_method(d, "Method")

        _save_bar_plot(
            d, "Method", "value",
            title=f"Exp3 Unified Compare - {metric_name}",
            y_label=metric_name,
            out_name=f"exp3_unified_{metric_name}.png",
            rotate=25
        )

    print("[ok] exp3 unified plots saved")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    plot_exp1_core_compare()
    plot_exp2_step_sweep()
    plot_exp3_algorithm_compare()
    plot_exp3_learned_realtime()
    plot_exp3_unified_if_possible()
    plot_train_history()

    print(f"Done. Plots are in: {OUT}")