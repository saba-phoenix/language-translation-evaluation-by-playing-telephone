import functools
import numpy as np
import pandas as pd
import scipy.stats
from mt_metrics_eval.data_new import EvalSet
from mt_metrics_eval.pce import compute_pairwise_p_values, compute_one_minus_pce

# eval set: evs

eval_sets = {
    "en-de": EvalSet(
        csv_path="benchmark_predictions/curated_merged.csv",
        lp="en-de",
        name="mqm",
    ),
    "en-ru": EvalSet(
        csv_path="benchmark_predictions/curated_merged.csv",
        lp="en-ru",
        name="mqm",
    ),
    "zh-en": EvalSet(
        csv_path="benchmark_predictions/curated_merged.csv",
        lp="zh-en",
        name="mqm",
    ),
}


def get_metric_scores(evs: EvalSet, metric: str) -> dict[str, list[float]]:
    """Get scores for a metric, filtering out bad systems"""
    scores_dict = evs.Scores("seg", metric)
    bad_systems = evs.outlier_sys_names | {evs.std_ref}
    return {
        system: scores
        for system, scores in scores_dict.items()
        if system not in bad_systems
    }


def calculate_spa(evs: EvalSet, metric: str) -> float:
    """Calculate SPA (1-PCE) for a given metric against human scores"""

    human_scores_dict = get_metric_scores(evs, "score")
    metric_scores_dict = get_metric_scores(evs, metric)

    common_systems = set(human_scores_dict.keys()) & set(metric_scores_dict.keys())
    common_systems = sorted(list(common_systems))

    if len(common_systems) < 2:
        print(f"Not enough systems for comparison: {len(common_systems)}")
        return np.nan

    num_segments = len(human_scores_dict[common_systems[0]])

    human_scores_array = np.array(
        [human_scores_dict[system] for system in common_systems], dtype=np.float32
    )

    metric_scores_array = np.array(
        [metric_scores_dict[system] for system in common_systems], dtype=np.float32
    )

    # handle None values by replacing with 0
    human_scores_array = np.nan_to_num(human_scores_array, nan=0.0)
    metric_scores_array = np.nan_to_num(metric_scores_array, nan=0.0)

    # compute pairwise p-values
    human_p_vals = compute_pairwise_p_values(human_scores_array)
    metric_p_vals = compute_pairwise_p_values(metric_scores_array)

    # calculate SPA (1-PCE)
    spa = compute_one_minus_pce(human_p_vals, metric_p_vals)

    return spa


def main():
    eval_sets = {
        "en-de": EvalSet(
            csv_path="benchmark_predictions/curated_merged.csv",
            lp="en-de",
            name="mqm",
        ),
        "en-ru": EvalSet(
            csv_path="benchmark_predictions/curated_merged.csv",
            lp="en-ru",
            name="mqm",
        ),
        "zh-en": EvalSet(
            csv_path="benchmark_predictions/curated_merged.csv",
            lp="zh-en",
            name="mqm",
        ),
    }

    metrics = [
        "xcomet",
        "cometda",
        "um_mr_hm",
        "um_mr_hm_mqm",
        "um_mr_rel",
        "reg_mr_hm",
        "reg_mr_rel",
        "xcomet_no_ref",
        "cometkiwi",
        "lr_ld",
        "lr_hd",
        "lr_ldd",
        "lr_hdd",
        "qe_mr",
        "qe_ft_mqm",
    ]

    all_data = []

    for metric in metrics:
        for lp, evs in eval_sets.items():
            print(f"Calculating SPA for {metric} in {lp}")
            spa = calculate_spa(evs, metric)
            all_data.append({"lp": lp, "metric": metric, "spa": spa})

    df = pd.DataFrame(all_data)

    print("\nSPA RESULTS")
    print(df.pivot_table(index="metric", columns="lp", values="spa"))

    pivot_df = df.pivot_table(index="metric", columns="lp", values="spa").reset_index()

    pivot_df[["en-de", "en-ru", "zh-en"]] = pivot_df[["en-de", "en-ru", "zh-en"]].round(
        3
    )

    pivot_df["avg_spa"] = pivot_df[["en-de", "en-ru", "zh-en"]].mean(axis=1).round(3)
    pivot_df["std_spa"] = pivot_df[["en-de", "en-ru", "zh-en"]].std(axis=1).round(3)

    pivot_df = pivot_df.sort_values("avg_spa", ascending=False).reset_index(drop=True)

    print("\nSPA AVERAGES ACROSS ALL LANGUAGE PAIRS")
    print(pivot_df)

    pivot_df.to_csv("benchmark_predictions/spa_results.csv", index=False)
    print("\nSaved SPA results to: benchmark_predictions/spa_results.csv")


if __name__ == "__main__":
    main()
