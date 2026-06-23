from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/agent_simulated_data.csv")
MODEL_PATH = Path("models/agent_models.joblib")
REPORT_DIR = Path("reports")
PLOT_DIR = Path("plots")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def did_from_means(df: pd.DataFrame, outcome_col: str) -> float:
    """
    Difference-in-differences estimate from raw means.
    """
    pivot = (
        df.groupby(["treatment_group", "post_period"])[outcome_col]
        .mean()
        .unstack()
        .sort_index()
    )

    # treatment_group = 1 minus treatment_group = 0
    post_effect_treat = pivot.loc[1, 1] - pivot.loc[1, 0]
    post_effect_control = pivot.loc[0, 1] - pivot.loc[0, 0]
    return post_effect_treat - post_effect_control

def summarize_group_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table by treatment and period.
    """
    summary = (
        df.groupby(["treatment_group", "post_period"])
        .agg(
            tasks=("task_id", "count"),
            avg_completion_time_min=("completion_time_min", "mean"),
            success_rate=("task_success", "mean"),
            avg_quality=("quality_score", "mean"),
            hallucination_rate=("hallucination_flag", "mean"),
            override_rate=("human_override_flag", "mean"),
            avg_revenue=("revenue_generated_usd", "mean"),
            avg_cost=("total_cost_usd", "mean"),
            avg_net_value=("net_value_usd", "mean"),
        )
        .reset_index()
    )

    summary["treatment_group"] = summary["treatment_group"].map({0: "Control", 1: "Treatment"})
    summary["post_period"] = summary["post_period"].map({0: "Pre", 1: "Post"})
    return summary

def model_effect(model, term="treatment_group:post_period"):
    """
    Pull coefficient, p-value, and 95% CI for the DiD term.
    """
    if term not in model.params.index:
        return None

    coef = model.params[term]
    pval = model.pvalues[term]
    ci_low, ci_high = model.conf_int().loc[term].tolist()

    return {
        "term": term,
        "coef": coef,
        "p_value": pval,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

def plot_kpis(df: pd.DataFrame):
    """
    Save a few simple charts for the repo.
    """
    chart_df = (
        df.groupby(["treatment_group", "post_period"])
        .agg(
            completion_time_min=("completion_time_min", "mean"),
            success_rate=("task_success", "mean"),
            hallucination_rate=("hallucination_flag", "mean"),
            net_value_usd=("net_value_usd", "mean"),
        )
        .reset_index()
    )

    chart_df["group"] = chart_df["treatment_group"].map({0: "Control", 1: "Treatment"})
    chart_df["period"] = chart_df["post_period"].map({0: "Pre", 1: "Post"})

    metrics = [
        ("completion_time_min", "Average Completion Time (min)", "completion_time_chart.png"),
        ("success_rate", "Task Success Rate", "success_rate_chart.png"),
        ("hallucination_rate", "Hallucination Rate", "hallucination_rate_chart.png"),
        ("net_value_usd", "Average Net Value (USD)", "net_value_chart.png"),
    ]

    for metric, title, filename in metrics:
        plt.figure(figsize=(8, 5))
        for group in ["Control", "Treatment"]:
            sub = chart_df[chart_df["group"] == group].sort_values("period")
            plt.plot(sub["period"], sub[metric], marker="o", label=group)

        plt.title(title)
        plt.xlabel("Period")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / filename, dpi=160)
        plt.close()

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    models = joblib.load(MODEL_PATH)

    # Actual metrics
    summary = summarize_group_kpis(df)
    summary_path = REPORT_DIR / "group_kpi_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Raw DID estimates from means
    did_completion = did_from_means(df, "completion_time_min")
    did_success = did_from_means(df, "task_success")
    did_quality = did_from_means(df, "quality_score")
    did_hallucination = did_from_means(df, "hallucination_flag")
    did_revenue = did_from_means(df, "revenue_generated_usd")
    did_cost = did_from_means(df, "total_cost_usd")
    did_net_value = did_from_means(df, "net_value_usd")

    # Model-based DiD effects
    effects = {}
    for name, model in models.items():
        effects[name] = model_effect(model)

    # Hallucination rate overall and by segment
    overall_hallucination_rate = df["hallucination_flag"].mean()
    hallucination_by_group = (
        df.groupby(["treatment_group", "post_period"])["hallucination_flag"]
        .mean()
        .reset_index()
    )
    hallucination_by_department = (
        df.groupby("department")["hallucination_flag"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Save report tables
    hallucination_by_group.to_csv(REPORT_DIR / "hallucination_by_group.csv", index=False)
    hallucination_by_department.to_csv(REPORT_DIR / "hallucination_by_department.csv", index=False)

    # Create charts
    plot_kpis(df)

    # Business interpretation
    avg_net_value_treat_post = df[
        (df["treatment_group"] == 1) & (df["post_period"] == 1)
    ]["net_value_usd"].mean()
    avg_net_value_control_post = df[
        (df["treatment_group"] == 0) & (df["post_period"] == 1)
    ]["net_value_usd"].mean()

    print("=" * 100)
    print("AI Agent Measurement Framework - Analysis")
    print("=" * 100)

    print("\n1) Summary table saved to:", summary_path)
    print(summary.to_string(index=False))

    print("\n2) Raw Difference-in-Differences estimates from means:")
    print(f"   Completion time (min): {did_completion:.3f}  (negative is good)")
    print(f"   Task success rate:      {did_success:.3f}  (positive is good)")
    print(f"   Quality score:          {did_quality:.3f}  (positive is good)")
    print(f"   Hallucination rate:     {did_hallucination:.3f}  (negative is good)")
    print(f"   Revenue (USD):          {did_revenue:.3f}  (positive is good)")
    print(f"   Cost (USD):             {did_cost:.3f}  (negative is good)")
    print(f"   Net value (USD):        {did_net_value:.3f}  (positive is good)")

    print("\n3) Model-based DiD coefficients:")
    for name, eff in effects.items():
        print(f"\n{name}:")
        if eff is None:
            print("   DiD term not found")
        else:
            print(
                f"   coef={eff['coef']:.4f}, p-value={eff['p_value']:.4f}, "
                f"95% CI=({eff['ci_low']:.4f}, {eff['ci_high']:.4f})"
            )

    print("\n4) Hallucination rate:")
    print(f"   Overall hallucination rate: {overall_hallucination_rate:.3%}")
    print("\n   By group / period:")
    print(hallucination_by_group.to_string(index=False))

    print("\n   By department:")
    print(hallucination_by_department.to_string(index=False))

    print("\n5) Post-rollout net value comparison:")
    print(f"   Treatment post-rollout avg net value: {avg_net_value_treat_post:.2f}")
    print(f"   Control post-rollout avg net value:   {avg_net_value_control_post:.2f}")

    print("\n6) Charts saved in:", PLOT_DIR.resolve())
    print("   - completion_time_chart.png")
    print("   - success_rate_chart.png")
    print("   - hallucination_rate_chart.png")
    print("   - net_value_chart.png")

if __name__ == "__main__":
    main()
