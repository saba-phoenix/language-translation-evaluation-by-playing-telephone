import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
from pipeline.evaluation_metrices.xcomet import evaluate as xcomet
from pipeline.eval_models.language_rotation_ld import evaluate as language_rotation_ld
from pipeline.eval_models.language_rotation_ldd import evaluate as language_rotation_ldd
from pipeline.eval_models.language_rotation_hdd import evaluate as language_rotation_hdd
from pipeline.eval_models.language_rotation_hd import evaluate as language_rotation_hd
from pipeline.eval_models.qe_model_rotation import evaluate as qe_model_rotation
from pipeline.eval_models.unifiedmetrics_model_rotation_hmref import (
    evaluate as unifiedmetrics_model_rotation_hmref,
)
from pipeline.eval_models.unifiedmetrics_model_rotation_relref import (
    evaluate as unifiedmetrics_model_rotation_relref,
)
from pipeline.eval_models.reg_model_rotation_hmref import (
    evaluate as reg_model_rotation_hmref,
)
from pipeline.eval_models.reg_model_rotation_relref import (
    evaluate as reg_model_rotation_relref,
)
from pipeline.eval_models.cometda import evaluate as cometda
from pipeline.eval_models.cometkiwi import evaluate as cometkiwi
from pipeline.eval_models.unifiedmetrics_model_rotation_hmref_ftmqm import (
    evaluate as unifiedmetrics_model_rotation_hmref_ftmqm,
)
from pipeline.eval_models.ablation_just_iteration import (
    evaluate as ablation_just_iteration,
)
from pipeline.eval_models.ablation_just_sentence import (
    evaluate as ablation_just_sentence,
)
from auc_comparison.prediction.eval import evaluate_translations
from pipeline.eval_models.ablation_asitis import evaluate as ablation_asitis

BATCH_SIZE = 128


def create_eval_wrapper(eval_func, use_ref: bool = False, batch_size: int = BATCH_SIZE):

    def wrapper(data: List[Dict[str, Any]]):
        if use_ref:
            formatted_data = [
                {
                    "id": item["id"],
                    "src": item["src"],
                    "mt": item["mt"],
                    "ref": item.get("ref", ""),
                }
                for item in data
            ]
        else:
            formatted_data = [
                {"id": item["id"], "src": item["src"], "mt": item["mt"]}
                for item in data
            ]
        return eval_func(formatted_data, batch_size=batch_size)

    return wrapper


evaluation_functions = [
    (create_eval_wrapper(xcomet, use_ref=False), "xcomet"),
    (
        create_eval_wrapper(unifiedmetrics_model_rotation_hmref_ftmqm, use_ref=False),
        "um_mr_hm_mqm",
    ),
    (create_eval_wrapper(language_rotation_ld, use_ref=False), "lr_ld"),
]


def process_multiple_csvs(
    input_csv_paths: List[str],
    evaluation_functions: List[Tuple[callable, str]],
    output_dir: str,
    batch_size: int = BATCH_SIZE,
    save_intermediate: bool = False,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for csv_path in input_csv_paths:
        print(f"Processing {csv_path}...")
        csv_name = Path(csv_path).stem

        # Generate output path for this file's results
        result_path = output_path / f"auc_results_ft_language_rotation.csv"

        # Process single CSV
        results_df = evaluate_translations(
            input_csv_path=csv_path,
            evaluation_functions=evaluation_functions,
            save_intermediate=save_intermediate,
            output_csv_path=str(result_path),
            batch_size=batch_size,
        )

        print(f"Results saved to {result_path}")


input_csv_paths = [
    # "outputs/comparison_dataset/commercial.csv",
    # "outputs/comparison_dataset/model_rotation.csv",
    "outputs/comparison_dataset/language_rotation.csv",
]

results_df = process_multiple_csvs(
    input_csv_paths=input_csv_paths,
    evaluation_functions=evaluation_functions,
    output_dir="auc_comparison/results",
    batch_size=BATCH_SIZE,
    save_intermediate=False,
)

print("\nResults Summary by Source File:")
print(results_df.groupby("source_file").describe())
