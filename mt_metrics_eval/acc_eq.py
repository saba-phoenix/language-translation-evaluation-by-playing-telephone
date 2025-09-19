# parts of the code are from https://github.com/google-research/mt-metrics-eval
import functools
import numpy as np
import pandas as pd
import scipy.stats
from mt_metrics_eval.data_new import EvalSet

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
    scores_dict = evs.Scores("seg", metric)
    # print("scores_dict", scores_dict)
    bad_systems = evs.outlier_sys_names | {evs.std_ref}
    return {
        system: scores
        for system, scores in scores_dict.items()
        if system not in bad_systems
    }


def kendall(x, y, variant):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError(
            "All inputs to `kendalltau` must be of the same "
            f"size, found x-size {x.size} and y-size {y.size}"
        )
    elif not x.size or not y.size:
        raise ValueError("x or y are empty")

    # check both x and y
    cnx = np.any(np.isnan(x))
    cny = np.any(np.isnan(y))
    contains_nan = cnx or cny
    if contains_nan:
        raise ValueError("x or y contains NaN")

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype("int64", copy=False)
        cnt = cnt[cnt > 1]
        return (
            (cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.0) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.0) * (2 * cnt + 5)).sum(),
        )

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind="mergesort")
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = scipy.stats._stats._kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype("int64", copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, _, _ = count_rank_tie(x)  # ties in x, stats
    ytie, _, _ = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2
    con = tot - ((xtie - ntie) + (ytie - ntie) + ntie + dis)

    tx = xtie - ntie
    ty = ytie - ntie
    txy = ntie

    if variant == "acc_eq":
        # Accuracy assuming tie optimization is done.
        return (con + txy) / (con + dis + tx + ty + txy), 0
    return 0


def calculate_correlations(
    evs: EvalSet,
    mqm_scores: dict[str, list[float]],
    metric_scores: dict[str, list[float]],
    coef: str,
) -> dict[str, float]:
    corr = evs.Correlation(mqm_scores, metric_scores)
    if coef == "accuracy-eq":
        corr_fn = functools.partial(corr.KendallWithTiesOpt, sample_rate=1.0)
    else:
        raise ValueError(coef)

    no_grouping = corr_fn()[0]
    group_by_item, _, num_items = corr_fn(average_by="item")
    return {
        "no_grouping": no_grouping,
        "group_by_item": group_by_item,
        "group_by_item_num_items": num_items,
    }


def analyze_ties(grouping: str, metric: str) -> pd.DataFrame:
    df = []
    for lp, evs in eval_sets.items():
        mqm_dict = get_metric_scores(evs, "mqm")

        scores_dict = get_metric_scores(evs, metric)
        num_translations = 0
        num_pairs = 0
        num_tied_pairs = 0
        num_zero_pairs = 0

        if grouping == "group_by_item":
            for i in range(len(evs.src)):
                item_scores = []
                for system, scores in scores_dict.items():
                    if scores[i] is not None and mqm_dict[system][i] is not None:
                        item_scores.append(scores[i])

                num_translations += len(item_scores)
                for j in range(len(item_scores)):
                    for k in range(j + 1, len(item_scores)):
                        num_pairs += 1
                        if item_scores[j] == item_scores[k]:
                            num_tied_pairs += 1
                            if item_scores[j] == 0.0:
                                num_zero_pairs += 1

        df.append(
            {
                "lp": lp,
                "num_translations": num_translations,
                "num_pairs": num_pairs,
                "num_tied_pairs": num_tied_pairs,
                "percent_of_pairs_tied": num_tied_pairs / num_pairs * 100,
                "num_zero_pairs": num_zero_pairs,
                "percent_of_tied_pairs_zero_tied": num_zero_pairs
                / num_tied_pairs
                * 100,
                "percent_of_pairs_zero_tied": num_zero_pairs / num_pairs * 100,
            }
        )
    return pd.DataFrame(df)


def main():
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
    coef = "accuracy-eq"
    groupings = ["group_by_item"]

    all_data = []
    for metric in metrics:
        for lp, evs in eval_sets.items():
            mqm_dict = get_metric_scores(evs, "score")
            # print("mqm_dict", mqm_dict)
            scores_dict = get_metric_scores(evs, metric)
            correlations = calculate_correlations(evs, mqm_dict, scores_dict, coef=coef)
            all_data.append(
                {
                    "lp": lp,
                    "metric": metric,
                    "grouping": groupings[0],
                    "correlation": correlations[groupings[0]],
                }
            )

    df = pd.DataFrame(all_data)
    print("\nCORRELATION RESULTS")
    print(df.pivot_table(index=["lp", "metric"], columns="grouping"))

    # NEW CODE: Calculate averages and save to CSV
    pivot_df = df.pivot_table(
        index="metric", columns="lp", values="correlation"
    ).reset_index()

    # Round language pair scores first to 3 decimal places
    pivot_df[["en-de", "en-ru", "zh-en"]] = pivot_df[["en-de", "en-ru", "zh-en"]].round(
        3
    )

    # Then calculate averages from the rounded values
    pivot_df["avg_correlation"] = (
        pivot_df[["en-de", "en-ru", "zh-en"]].mean(axis=1).round(3)
    )
    pivot_df["std_correlation"] = (
        pivot_df[["en-de", "en-ru", "zh-en"]].std(axis=1).round(3)
    )

    pivot_df = pivot_df.sort_values("avg_correlation", ascending=False).reset_index(
        drop=True
    )

    print("\nAVERAGE CORRELATIONS ACROSS ALL LANGUAGE PAIRS")
    print(pivot_df)

    pivot_df.to_csv(
        "benchmark_predictions/acc_eq_average_correlations.csv", index=False
    )
    print("\nSaved averages to: benchmark_predictions/acc_eq_average_correlations.csv")


if __name__ == "__main__":
    main()
