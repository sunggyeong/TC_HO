# plot_exp1_exp2_py.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def find_col(df: pd.DataFrame, aliases):
    norm_map = {_normalize(c): c for c in df.columns}
    for a in aliases:
        key = _normalize(a)
        if key in norm_map:
            return norm_map[key]
    return None


def load_csv(path: Path):
    if not path.exists():
        print(f"[WARN] 파일 없음: {path}")
        return None
    df = pd.read_csv(path)
    print(f"[LOAD] {path} shape={df.shape}")
    print(f"       columns={list(df.columns)}")
    return df


def save_and_show(fig, save_path: Path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {save_path}")
    plt.show()


def metric_bar(df, method_col, metric_col, title, ylabel, save_name):
    fig = plt.figure(figsize=(8.5, 4.8))
    plt.bar(df[method_col].astype(str), df[metric_col])
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    save_and_show(fig, PLOTS_DIR / save_name)


def metric_line(df, x_col, y_col, title, ylabel, save_name, group_col=None):
    fig = plt.figure(figsize=(8.8, 5.0))
    if group_col and group_col in df.columns:
        for g, d in df.groupby(group_col):
            d = d.sort_values(x_col)
            plt.plot(d[x_col], d[y_col], marker="o", label=str(g))
        plt.legend()
    else:
        d = df.sort_values(x_col)
        plt.plot(d[x_col], d[y_col], marker="o")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    save_and_show(fig, PLOTS_DIR / save_name)


# --------------------------
# Exp1: 핵심 비교
# --------------------------
def plot_exp1():
    path = RESULTS_DIR / "exp1_core_compare_summary.csv"
    df = load_csv(path)
    if df is None:
        return

    method_col = find_col(df, ["Method", "method", "algorithm", "policy", "scheme"])
    if method_col is None:
        print("[WARN] Exp1: method 컬럼을 찾지 못함")
        return

    # 네 실제 summary 컬럼명에 맞춘 alias
    metrics = [
        ("Effective QoE", ["EffectiveQoE_mean"], "Exp1 - Effective QoE", "QoE", "exp1_effective_qoe.png"),
        ("Availability", ["Availability_mean"], "Exp1 - Availability", "Availability", "exp1_availability.png"),
        ("Mean Latency (ms)", ["MeanLatency_ms_mean"], "Exp1 - Mean Latency", "Latency (ms)", "exp1_mean_latency_ms.png"),
        ("P95 Latency (ms)", ["P95Latency_ms_mean"], "Exp1 - P95 Latency", "P95 Latency (ms)", "exp1_p95_latency_ms.png"),
        ("Mean Interruption (ms)", ["MeanInterruption_ms_mean"], "Exp1 - Mean Interruption", "Interruption (ms)", "exp1_mean_interruption_ms.png"),
        ("P95 Interruption (ms)", ["P95Interruption_ms_mean"], "Exp1 - P95 Interruption", "P95 Interruption (ms)", "exp1_p95_interruption_ms.png"),
        ("Deadline Miss Ratio", ["DeadlineMissRatio_mean"], "Exp1 - Deadline Miss Ratio", "Ratio", "exp1_deadline_miss_ratio.png"),
        ("HO Failure Ratio", ["HO_Failure_Ratio_mean"], "Exp1 - HO Failure Ratio", "Ratio", "exp1_ho_failure_ratio.png"),
        ("Decision Completion Rate", ["DecisionCompletionRate_mean"], "Exp1 - Decision Completion Rate", "Rate", "exp1_decision_completion_rate.png"),
        ("HO Attempt Count", ["HO_Attempt_Count_mean"], "Exp1 - HO Attempt Count", "Count", "exp1_ho_attempt_count.png"),
        ("PingPong Ratio", ["PingPong_Ratio_mean"], "Exp1 - Ping-Pong Ratio", "Ratio", "exp1_pingpong_ratio.png"),
    ]

    found_any = False
    for _, aliases, title, ylabel, save_name in metrics:
        y_col = find_col(df, aliases)
        if y_col is None:
            continue
        found_any = True
        metric_bar(df, method_col, y_col, title, ylabel, save_name)

    if not found_any:
        print("[WARN] Exp1: 그릴 수 있는 metric 컬럼을 못 찾음")


# --------------------------
# Exp2: step sweep
# --------------------------
def plot_exp2():
    path = RESULTS_DIR / "exp2_step_sweep_summary.csv"
    df = load_csv(path)
    if df is None:
        return

    x_col = find_col(df, ["SamplingSteps", "steps", "num_steps", "sample_steps", "k"])
    group_col = find_col(df, ["Method", "Mode", "method", "algorithm", "policy", "sampler", "model"])

    if x_col is None:
        print("[WARN] Exp2: step 축 컬럼을 찾지 못함")
        return

    metrics = [
        ("Effective QoE", ["EffectiveQoE_mean"], "Exp2 Step Sweep - Effective QoE", "QoE", "exp2_effective_qoe.png"),
        ("Availability", ["Availability_mean"], "Exp2 Step Sweep - Availability", "Availability", "exp2_availability.png"),
        ("Mean Latency (ms)", ["MeanLatency_ms_mean"], "Exp2 Step Sweep - Mean Latency", "Latency (ms)", "exp2_mean_latency_ms.png"),
        ("Mean Interruption (ms)", ["MeanInterruption_ms_mean"], "Exp2 Step Sweep - Mean Interruption", "Interruption (ms)", "exp2_mean_interruption_ms.png"),
        ("Deadline Miss Ratio", ["DeadlineMissRatio_mean"], "Exp2 Step Sweep - Deadline Miss Ratio", "Ratio", "exp2_deadline_miss_ratio.png"),
        ("HO Failure Ratio", ["HO_Failure_Ratio_mean"], "Exp2 Step Sweep - HO Failure Ratio", "Ratio", "exp2_ho_failure_ratio.png"),
        ("Decision Completion Rate", ["DecisionCompletionRate_mean"], "Exp2 Step Sweep - Decision Completion Rate", "Rate", "exp2_decision_completion_rate.png"),
    ]

    found_any = False
    for _, aliases, title, ylabel, save_name in metrics:
        y_col = find_col(df, aliases)
        if y_col is None:
            continue
        found_any = True
        metric_line(df, x_col, y_col, title, ylabel, save_name, group_col=group_col)

    if not found_any:
        print("[WARN] Exp2: 그릴 수 있는 metric 컬럼을 못 찾음")


# --------------------------
# Timeline 비교 (네 실제 timeline 컬럼 기반)
# --------------------------
def plot_timeline_compare():
    path_cons = RESULTS_DIR / "timeline_predicted_consistency_seed0.csv"
    path_diff = RESULTS_DIR / "timeline_predicted_diffusion_seed0.csv"

    df_c = load_csv(path_cons)
    df_d = load_csv(path_diff)
    if df_c is None or df_d is None:
        return

    # 시간축: sim_time_sec 우선, 없으면 t_idx
    x_col_c = find_col(df_c, ["sim_time_sec", "t_idx", "t", "time", "slot"])
    x_col_d = find_col(df_d, ["sim_time_sec", "t_idx", "t", "time", "slot"])
    if x_col_c is None or x_col_d is None:
        print("[WARN] Timeline: 시간축 컬럼을 찾지 못함")
        return

    # 1) latency
    lat_c = find_col(df_c, ["latency_ms"])
    lat_d = find_col(df_d, ["latency_ms"])
    if lat_c and lat_d:
        fig = plt.figure(figsize=(10.5, 4.8))
        plt.plot(df_d[x_col_d], df_d[lat_d], label="Predicted + Diffusion")
        plt.plot(df_c[x_col_c], df_c[lat_c], label="Predicted + Consistency")
        plt.title("Timeline - Link Latency")
        plt.xlabel("Time (s)" if "sec" in x_col_c.lower() else "Time Slot")
        plt.ylabel("Latency (ms)")
        plt.grid(alpha=0.3)
        plt.legend()
        save_and_show(fig, PLOTS_DIR / "timeline_latency_compare.png")

    # 2) interruption
    intr_c = find_col(df_c, ["interruption_ms"])
    intr_d = find_col(df_d, ["interruption_ms"])
    if intr_c and intr_d:
        fig = plt.figure(figsize=(10.5, 4.8))
        plt.plot(df_d[x_col_d], df_d[intr_d], label="Predicted + Diffusion")
        plt.plot(df_c[x_col_c], df_c[intr_c], label="Predicted + Consistency")
        plt.title("Timeline - Service Interruption")
        plt.xlabel("Time (s)" if "sec" in x_col_c.lower() else "Time Slot")
        plt.ylabel("Interruption (ms)")
        plt.grid(alpha=0.3)
        plt.legend()
        save_and_show(fig, PLOTS_DIR / "timeline_interruption_compare.png")

    # 3) inference latency (핵심!)
    inf_c = find_col(df_c, ["inference_latency_ms"])
    inf_d = find_col(df_d, ["inference_latency_ms"])
    if inf_c and inf_d:
        fig = plt.figure(figsize=(10.5, 4.8))
        plt.plot(df_d[x_col_d], df_d[inf_d], label="Predicted + Diffusion")
        plt.plot(df_c[x_col_c], df_c[inf_c], label="Predicted + Consistency")
        plt.title("Timeline - Inference Latency (Core Claim)")
        plt.xlabel("Time (s)" if "sec" in x_col_c.lower() else "Time Slot")
        plt.ylabel("Inference Latency (ms)")
        plt.grid(alpha=0.3)
        plt.legend()
        save_and_show(fig, PLOTS_DIR / "timeline_inference_latency_compare.png")

    # 4) outage flag
    out_c = find_col(df_c, ["outage"])
    out_d = find_col(df_d, ["outage"])
    if out_c and out_d:
        fig = plt.figure(figsize=(10.5, 4.2))
        plt.step(df_d[x_col_d], df_d[out_d], where="post", label="Predicted + Diffusion")
        plt.step(df_c[x_col_c], df_c[out_c], where="post", label="Predicted + Consistency")
        plt.title("Timeline - Outage Flag")
        plt.xlabel("Time (s)" if "sec" in x_col_c.lower() else "Time Slot")
        plt.ylabel("Outage (0/1)")
        plt.yticks([0, 1])
        plt.grid(alpha=0.3)
        plt.legend()
        save_and_show(fig, PLOTS_DIR / "timeline_outage_compare.png")

    # 5) deadline miss flag (핵심!)
    dm_c = find_col(df_c, ["deadline_miss"])
    dm_d = find_col(df_d, ["deadline_miss"])
    if dm_c and dm_d:
        fig = plt.figure(figsize=(10.5, 4.2))
        plt.step(df_d[x_col_d], df_d[dm_d], where="post", label="Predicted + Diffusion")
        plt.step(df_c[x_col_c], df_c[dm_c], where="post", label="Predicted + Consistency")
        plt.title("Timeline - Deadline Miss Flag (Core Claim)")
        plt.xlabel("Time (s)" if "sec" in x_col_c.lower() else "Time Slot")
        plt.ylabel("Deadline Miss (0/1)")
        plt.yticks([0, 1])
        plt.grid(alpha=0.3)
        plt.legend()
        save_and_show(fig, PLOTS_DIR / "timeline_deadline_miss_compare.png")

    # 6) HO attempt / success / failure
    for col_name, title, save_name in [
        ("ho_attempt", "Timeline - HO Attempt", "timeline_ho_attempt_compare.png"),
        ("ho_success", "Timeline - HO Success", "timeline_ho_success_compare.png"),
        ("ho_failure", "Timeline - HO Failure", "timeline_ho_failure_compare.png"),
    ]:
        c_col = find_col(df_c, [col_name])
        d_col = find_col(df_d, [col_name])
        if c_col and d_col:
            fig = plt.figure(figsize=(10.5, 4.2))
            plt.step(df_d[x_col_d], df_d[d_col], where="post", label="Predicted + Diffusion")
            plt.step(df_c[x_col_c], df_c[c_col], where="post", label="Predicted + Consistency")
            plt.title(title)
            plt.xlabel("Time (s)" if "sec" in x_col_c.lower() else "Time Slot")
            plt.ylabel("Flag (0/1)")
            plt.yticks([0, 1])
            plt.grid(alpha=0.3)
            plt.legend()
            save_and_show(fig, PLOTS_DIR / save_name)


def main():
    print("=== Plotting results with matplotlib ===")
    print(f"RESULTS_DIR = {RESULTS_DIR.resolve()}")
    plot_exp1()
    plot_exp2()
    plot_timeline_compare()
    print("Done.")


if __name__ == "__main__":
    main()