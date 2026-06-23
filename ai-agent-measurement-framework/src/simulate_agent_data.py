from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

# -----------------------------
# Config
# -----------------------------
SEED = 42
N_ROWS = 12000
ROLLOUT_DATE = pd.Timestamp("2025-04-01")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# -----------------------------
# Helpers
# -----------------------------
def sigmoid(x):
    return expit(x)

def clip_series(x, lower, upper):
    return np.clip(x, lower, upper)

# -----------------------------
# Simulation
# -----------------------------
def generate_simulated_agent_data(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments = ["Support", "Sales", "Marketing", "Research"]
    task_types = ["simple", "medium", "complex"]

    dept_probs = [0.35, 0.20, 0.25, 0.20]
    task_probs = [0.45, 0.35, 0.20]

    # Users and teams
    n_users = 600
    n_teams = 30

    user_ids = rng.integers(1000, 1000 + n_users, size=n_rows)
    team_ids = rng.integers(1, n_teams + 1, size=n_rows)

    # Team-level treatment assignment (cleaner for causal analysis)
    team_treatment_map = {
        team_id: int(rng.random() < 0.5) for team_id in range(1, n_teams + 1)
    }

    # Dates across a rollout window
    dates = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    task_dates = rng.choice(dates, size=n_rows, replace=True)

    df = pd.DataFrame(
        {
            "task_id": np.arange(1, n_rows + 1),
            "user_id": user_ids,
            "team_id": team_ids,
            "date": pd.to_datetime(task_dates),
            "department": rng.choice(departments, size=n_rows, p=dept_probs),
            "task_type": rng.choice(task_types, size=n_rows, p=task_probs),
        }
    )

    # Treatment group is assigned at team level
    df["treatment_group"] = df["team_id"].map(team_treatment_map).astype(int)

    # Post-rollout period
    df["post_period"] = (df["date"] >= ROLLOUT_DATE).astype(int)

    # Agent is actually used only if the team has access AND it is post-rollout.
    # Adoption is not perfect.
    adoption_prob = []
    for _, row in df.iterrows():
        base = 0.00
        if row["treatment_group"] == 1 and row["post_period"] == 1:
            base = 0.75
            if row["task_type"] == "complex":
                base -= 0.08
            if row["department"] == "Marketing":
                base += 0.05
            if row["department"] == "Research":
                base -= 0.05
        adoption_prob.append(np.clip(base, 0.0, 0.95))

    df["agent_used"] = rng.binomial(1, adoption_prob)

    # Complexity score: simple=1, medium=2, complex=3 with a small jitter
    complexity_map = {"simple": 1.0, "medium": 2.0, "complex": 3.0}
    df["task_complexity"] = df["task_type"].map(complexity_map).astype(float)
    df["task_complexity"] = df["task_complexity"] + rng.normal(0, 0.12, size=n_rows)
    df["task_complexity"] = clip_series(df["task_complexity"], 1.0, 3.4)

    # Department effects
    dept_time_effect = {
        "Support": 2.5,
        "Sales": 1.5,
        "Marketing": 0.8,
        "Research": 3.0,
    }
    dept_quality_effect = {
        "Support": 0.05,
        "Sales": 0.00,
        "Marketing": 0.12,
        "Research": -0.06,
    }
    dept_hallucination_effect = {
        "Support": -0.10,
        "Sales": 0.00,
        "Marketing": 0.04,
        "Research": 0.15,
    }

    df["dept_time_effect"] = df["department"].map(dept_time_effect)
    df["dept_quality_effect"] = df["department"].map(dept_quality_effect)
    df["dept_hallucination_effect"] = df["department"].map(dept_hallucination_effect)

    # Task-specific base values
    task_value_map = {"simple": 35, "medium": 70, "complex": 120}
    df["task_value"] = df["task_type"].map(task_value_map).astype(float)

    # -----------------------------
    # Outcomes
    # -----------------------------
    # Completion time in minutes
    base_time = (
        9.0
        + 7.5 * df["task_complexity"]
        + df["dept_time_effect"]
        + rng.normal(0, 2.0, size=n_rows)
    )

    # Agent effect reduces time
    agent_time_multiplier = np.where(
        df["agent_used"] == 1,
        np.where(df["task_type"] == "complex", 0.82, 0.74),
        1.0,
    )

    df["completion_time_min"] = np.maximum(
        1.0, base_time * agent_time_multiplier
    )

    # Hallucination probability
    hall_logit = (
        -3.0
        + 0.95 * df["agent_used"]
        + 0.75 * (df["task_complexity"] - 1.0)
        + df["dept_hallucination_effect"]
        + np.where(df["task_type"] == "complex", 0.30, 0.0)
    )
    df["hallucination_flag"] = rng.binomial(1, sigmoid(hall_logit))

    # Quality score (1 to 5)
    raw_quality = (
        4.25
        - 0.38 * (df["task_complexity"] - 1.0)
        + 0.20 * df["agent_used"]
        + df["dept_quality_effect"]
        - 0.75 * df["hallucination_flag"]
        + rng.normal(0, 0.18, size=n_rows)
    )
    df["quality_score"] = clip_series(raw_quality, 1.0, 5.0)

    # Task success probability
    success_logit = (
        1.35
        + 0.45 * df["agent_used"]
        - 0.70 * (df["task_complexity"] - 1.0)
        - 1.10 * df["hallucination_flag"]
        + 0.20 * df["dept_quality_effect"]
    )
    df["task_success"] = rng.binomial(1, sigmoid(success_logit))

    # Human override more likely when hallucination happens or quality is low
    override_logit = (
        -1.8
        + 1.35 * df["hallucination_flag"]
        - 0.75 * (df["quality_score"] - 3.0)
        + 0.25 * df["agent_used"]
    )
    df["human_override_flag"] = rng.binomial(1, sigmoid(override_logit))

    # Cost model: human labor + AI cost + override cost
    hourly_human_cost = 45.0
    ai_cost_per_task = 0.12
    override_cost = 1.50

    df["human_cost_usd"] = (df["completion_time_min"] / 60.0) * hourly_human_cost
    df["ai_cost_usd"] = df["agent_used"] * ai_cost_per_task * (
        1.0 + 0.08 * (df["task_complexity"] - 1.0)
    )
    df["override_cost_usd"] = df["human_override_flag"] * override_cost
    df["total_cost_usd"] = (
        df["human_cost_usd"] + df["ai_cost_usd"] + df["override_cost_usd"]
    )

    # Revenue generated: only successful tasks create value
    # Agent gives a small lift on successful tasks, but not huge
    revenue_lift = np.where(df["agent_used"] == 1, 1.08, 1.00)
    df["revenue_generated_usd"] = (
        df["task_value"] * df["task_success"] * revenue_lift
        + rng.normal(0, 2.5, size=n_rows)
    )
    df["revenue_generated_usd"] = np.maximum(0.0, df["revenue_generated_usd"])

    # Net value
    df["net_value_usd"] = df["revenue_generated_usd"] - df["total_cost_usd"]

    # Helpful time buckets
    df["week"] = df["date"].dt.to_period("W").astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # Final cleanup
    keep_cols = [
        "task_id",
        "user_id",
        "team_id",
        "date",
        "week",
        "month",
        "department",
        "task_type",
        "task_complexity",
        "treatment_group",
        "post_period",
        "agent_used",
        "completion_time_min",
        "task_success",
        "quality_score",
        "hallucination_flag",
        "human_override_flag",
        "human_cost_usd",
        "ai_cost_usd",
        "override_cost_usd",
        "total_cost_usd",
        "task_value",
        "revenue_generated_usd",
        "net_value_usd",
    ]

    return df[keep_cols].sort_values("date").reset_index(drop=True)

def main():
    df = generate_simulated_agent_data()
    out_path = PROCESSED_DIR / "agent_simulated_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df.head(10).to_string(index=False))
    print("\nSummary:")
    print(df[["agent_used", "task_success", "hallucination_flag", "quality_score"]].describe())

if __name__ == "__main__":
    main()
