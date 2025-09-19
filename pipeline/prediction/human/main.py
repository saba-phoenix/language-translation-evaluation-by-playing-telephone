import pandas as pd
from typing import List, Dict, Any
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
from pipeline.prediction.human.eval import evaluate_translations
from pipeline.eval_models.qe_ft_mqm import evaluate as qe_ft_mqm

# Global configuration
BATCH_SIZE = 128  # Unified batch size for all models


def create_eval_wrapper(eval_func, use_ref: bool = False, batch_size: int = BATCH_SIZE):

    def wrapper(data: List[Dict[str, Any]]):
        if use_ref:
            # For models that need reference text
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
            # For models that only need source and machine translation
            formatted_data = [
                {"id": item["id"], "src": item["src"], "mt": item["mt"]}
                for item in data
            ]
        return eval_func(formatted_data, batch_size=batch_size)

    return wrapper


evaluation_functions = [
    (create_eval_wrapper(xcomet, use_ref=True), "xcomet"),
    (create_eval_wrapper(cometda, use_ref=True), "cometda"),
    (
        create_eval_wrapper(unifiedmetrics_model_rotation_hmref, use_ref=True),
        "um_mr_hm",
    ),
    (
        create_eval_wrapper(unifiedmetrics_model_rotation_hmref_ftmqm, use_ref=True),
        "um_mr_hm_mqm",
    ),
    (
        create_eval_wrapper(unifiedmetrics_model_rotation_relref, use_ref=True),
        "um_mr_rel",
    ),
    (create_eval_wrapper(reg_model_rotation_hmref, use_ref=True), "reg_mr_hm"),
    (create_eval_wrapper(reg_model_rotation_relref, use_ref=True), "reg_mr_rel"),
    (create_eval_wrapper(xcomet, use_ref=False), "xcomet_no_ref"),
    (create_eval_wrapper(cometkiwi, use_ref=False), "cometkiwi"),
    (create_eval_wrapper(language_rotation_ld, use_ref=False), "lr_ld"),
    (create_eval_wrapper(language_rotation_hd, use_ref=False), "lr_hd"),
    (create_eval_wrapper(language_rotation_ldd, use_ref=False), "lr_ldd"),
    (create_eval_wrapper(language_rotation_hdd, use_ref=False), "lr_hdd"),
    (create_eval_wrapper(qe_model_rotation, use_ref=False), "qe_mr"),
    (create_eval_wrapper(qe_ft_mqm, use_ref=False), "qe_ft_mqm"),
]


results_df = evaluate_translations(
    input_csv_path="benchmark_predictions/mqm_6.csv",
    evaluation_functions=evaluation_functions,
    save_intermediate=False,
    output_csv_path="benchmark_predictions/mqm_7.csv",
    batch_size=BATCH_SIZE,
)

print(results_df.head())
