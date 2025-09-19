import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def generate_latex_table(df, output_dir):
    comparisons = df.columns.levels[0]
    score_types = df.columns.levels[1]

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{AUC Scores for Different Model Comparisons and Metrics}")
    latex.append("  \\label{tab:auc_scores}")

    num_cols = 1 + len(score_types) * len(comparisons)
    latex.append(f"  \\begin{{tabular}}{{l{' c' * (num_cols-1)}}}")

    latex.append("    \\toprule")

    header_row = ["\\multirow{2}{*}{Model}"]
    for comp in comparisons:
        header_row.append(f"\\multicolumn{{{len(score_types)}}}{{c}}{{{comp}}}")
    latex.append("    " + " & ".join(header_row) + " \\\\")

    score_headers = []
    for _ in comparisons:
        for score in score_types:
            score_headers.append(score)
    latex.append("    " + " & " + " & ".join(score_headers) + " \\\\")
    latex.append("    \\midrule")

    for model in df.index:
        row = [model]
        for comp in comparisons:
            for score in score_types:
                value = df.loc[model, (comp, score)]
                if pd.isna(value):
                    row.append("-")
                else:
                    row.append(f"{value:.3f}")
        latex.append("    " + " & ".join(row) + " \\\\")

    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    with open(output_dir / "combined_auc_scores_latex.txt", "w") as f:
        f.write(latex_str)

    return latex_str


def calculate_auc(scores1, scores2):
    """
    Calculate AUC score for comparing two sets of scores using raw differences.
    The ROC curve will show discrimination across all possible thresholds.

    For each pair of scores:
    - Randomly decide whether to keep original order (heads) or swap (tails)
    - If heads: keep original difference, y_true = 1
    - If tails: negate difference, y_true = 0
    This creates a balanced dataset for AUC calculation.
    """
    score_diffs = scores1 - scores2

    np.random.seed(42)
    coin_flips = np.random.randint(0, 2, size=len(score_diffs))

    modified_diffs = np.zeros_like(score_diffs, dtype=float)
    y_true = np.zeros_like(score_diffs, dtype=int)

    for i in range(len(score_diffs)):
        if coin_flips[i] == 1:
            modified_diffs[i] = score_diffs[i]
            y_true[i] = 1
        else:
            modified_diffs[i] = -score_diffs[i]
            y_true[i] = 0

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=modified_diffs)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def plot_roc_curves(
    fpr_dict, tpr_dict, auc_dict, comparison_name, output_dir, score_type
):
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    for model, fpr in fpr_dict.items():
        plt.plot(
            fpr,
            tpr_dict[model],
            label=f"{model} (AUC = {auc_dict[model]:.3f})",
            linewidth=2,
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves for {comparison_name} ({score_type})")
    plt.legend(loc="lower right")

    plt.savefig(
        output_dir / f"roc_curve_{score_type}_{comparison_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_tables(project_name="full_noref", score_types=None):

    data_dir = Path(f"auc_comparison/results/")
    file_path = data_dir / f"{project_name}.csv"

    comparisons = {"1_vs_3": (1, 3), "2_vs_3": (2, 3), "1_vs_2": (1, 2)}

    model_iterations = {
        "First Language": [2, 8, 14],
        "Second Language": [4, 10, 16],
        "Third Language": [6, 12, 18],
    }

    all_results = {model: {} for model in model_iterations.keys()}

    df = pd.read_csv(file_path)
    df["main_id"] = df["id"].apply(lambda x: "_".join(x.split("_")[1:-1]))
    df["iteration"] = df["id"].apply(lambda x: int(x.split("_")[-1]))

    plot_dir = data_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for full_score_type, short_name in score_types.items():
        auc_results = {
            comp: {"fpr": {}, "tpr": {}, "auc": {}} for comp in comparisons.keys()
        }

        for model, iterations in model_iterations.items():
            model_df = df[df["iteration"].isin(iterations)].copy()

            iteration_mapping = {iterations[0]: 1, iterations[1]: 2, iterations[2]: 3}
            model_df["relative_iteration"] = model_df["iteration"].map(
                iteration_mapping
            )

            pivot = model_df.pivot_table(
                index="main_id",
                columns="relative_iteration",
                values=full_score_type,
            )

            pivot = pivot.dropna(subset=[1, 2, 3])

            for comp_name, (iter1, iter2) in comparisons.items():
                fpr, tpr, roc_auc = calculate_auc(
                    pivot[iter1].values, pivot[iter2].values
                )

                auc_results[comp_name]["fpr"][model] = fpr
                auc_results[comp_name]["tpr"][model] = tpr
                auc_results[comp_name]["auc"][model] = roc_auc

                if comp_name not in all_results[model]:
                    all_results[model][comp_name] = {}
                all_results[model][comp_name][short_name] = round(roc_auc, 3)

        for comp_name in comparisons.keys():
            plot_roc_curves(
                auc_results[comp_name]["fpr"],
                auc_results[comp_name]["tpr"],
                auc_results[comp_name]["auc"],
                comp_name,
                plot_dir,
                full_score_type,
            )

        results_rows = []
        for model in model_iterations.keys():
            row = {"model": model}
            for comp_name in comparisons.keys():
                if model in auc_results[comp_name]["auc"]:
                    row[f"{comp_name}_auc"] = round(
                        auc_results[comp_name]["auc"][model], 3
                    )
            results_rows.append(row)

        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(
            data_dir / f"auc_tables/auc_scores_{project_name}_{full_score_type}.csv",
            index=False,
        )

    columns = pd.MultiIndex.from_product(
        [list(comparisons.keys()), list(score_types.values())]
    )

    final_table = pd.DataFrame(index=list(model_iterations.keys()), columns=columns)

    for model, comp_data in all_results.items():
        for comp_name, score_data in comp_data.items():
            for score_short, value in score_data.items():
                final_table.loc[model, (comp_name, score_short)] = value

    final_table.to_csv(data_dir / "combined_auc_scores.csv")

    latex_table = generate_latex_table(final_table, data_dir)
    with open(data_dir / "combined_auc_scores_latex.txt", "w") as f:
        f.write(latex_table)

    print("Results saved successfully!")
    print(f"ROC curves saved in {plot_dir}")
    print(
        f"Combined results table saved as combined_auc_scores.csv and combined_auc_scores.xlsx"
    )
    print(f"LaTeX table saved as combined_auc_scores_latex.txt")

    print("\nCombined AUC Scores Table:")
    print(final_table)


if __name__ == "__main__":
    project_name = "language_rotation"
    score_types = {"xcomet_no_ref": "xcomet", "lr_ld": "lr", "qe_mr": "mr"}
    create_tables(project_name, score_types)
