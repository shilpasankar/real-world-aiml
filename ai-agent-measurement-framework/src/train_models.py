from pathlib import Path
import joblib
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = Path("data/processed/agent_simulated_data.csv")
MODEL_PATH = Path("models/agent_models.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def fit_models(df: pd.DataFrame):
    """
    Fit a set of models:
      - completion time (OLS)
      - task success (Binomial GLM)
      - quality score (OLS)
      - hallucination (Binomial GLM)
      - revenue generated (OLS)
      - net value (OLS)

    The key causal term is treatment_group * post_period.
    """
    # Make sure binary columns are numeric ints
    for col in ["treatment_group", "post_period", "agent_used", "task_success", "hallucination_flag", "human_override_flag"]:
        df[col] = df[col].astype(int)

    formula_common = "treatment_group * post_period + task_complexity + C(task_type) + C(department)"

    completion_model = smf.ols(
        formula=f"completion_time_min ~ {formula_common}",
        data=df
    ).fit(cov_type="HC3")

    success_model = smf.glm(
        formula=f"task_success ~ {formula_common}",
        data=df,
        family=sm.families.Binomial()
    ).fit()

    quality_model = smf.ols(
        formula=f"quality_score ~ {formula_common}",
        data=df
    ).fit(cov_type="HC3")

    hallucination_model = smf.glm(
        formula=f"hallucination_flag ~ {formula_common}",
        data=df,
        family=sm.families.Binomial()
    ).fit()

    revenue_model = smf.ols(
        formula=f"revenue_generated_usd ~ {formula_common}",
        data=df
    ).fit(cov_type="HC3")

    net_value_model = smf.ols(
        formula=f"net_value_usd ~ {formula_common}",
        data=df
    ).fit(cov_type="HC3")

    models = {
        "completion_model": completion_model,
        "success_model": success_model,
        "quality_model": quality_model,
        "hallucination_model": hallucination_model,
        "revenue_model": revenue_model,
        "net_value_model": net_value_model,
    }

    return models

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    models = fit_models(df)
    joblib.dump(models, MODEL_PATH)

    print(f"Saved models to: {MODEL_PATH}\n")

    for name, model in models.items():
        print("=" * 90)
        print(name)
        print(model.summary())
        print("\n")

if __name__ == "__main__":
    main()
