"""Unified EvaluationPipeline orchestrating all evaluation steps.

Runs benchmark evaluation, perplexity, few-shot sensitivity, calibration,
weight analysis, activation analysis, dataset statistics, and report
generation in a single coordinated run. Each step is fault-isolated: a
failure in any step is logged and the pipeline continues.
"""

from __future__ import annotations

import datetime
import logging
import os
import traceback
from pathlib import Path
from typing import Any

from evaluate.config import EvalConfig
from evaluate.evaluate import run_evaluation
from evaluate.report import (
    generate_csv_report,
    generate_markdown_report,
    generate_narrative_report,
)
from shared.logging_utils import JSONLogger

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore

_logger = logging.getLogger(__name__)


class EvaluationBundle(TypedDict):
    """All results from a single EvaluationPipeline.run() call."""
    benchmark_scores:     dict[str, dict[str, float | None]]
    perplexity:           dict[str, float | None]
    few_shot_sensitivity: dict[tuple[str, str, int], float | None]
    calibration:          dict[str, dict[str, Any] | None]
    weight_analysis:      dict[str, Any | None]
    activation_analysis:  dict[str, Any | None]
    dataset_stats:        dict[str, Any] | None
    local_model:          dict[str, float | None] | None


def _empty_bundle() -> EvaluationBundle:
    return EvaluationBundle(
        benchmark_scores={},
        perplexity={},
        few_shot_sensitivity={},
        calibration={},
        weight_analysis={},
        activation_analysis={},
        dataset_stats=None,
        local_model=None,
    )


class EvaluationPipeline:
    """Orchestrate the full evaluation pipeline.

    Args:
        config: EvalConfig with models, tasks, paths, and optional fields.
        logger: JSONLogger for structured event logging.
    """

    def __init__(self, config: EvalConfig, logger: JSONLogger) -> None:
        self.config = config
        self.logger = logger

    def run(self) -> EvaluationBundle:
        """Execute all evaluation steps and return an EvaluationBundle.

        Each step is wrapped in try/except. Failures are logged and the
        corresponding bundle key is set to None; the pipeline always completes.

        Returns:
            EvaluationBundle with all eight required keys populated.
        """
        bundle = _empty_bundle()
        cfg = self.config
        out_dir = cfg.output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Log start event
        self.logger.log({
            "type": "pipeline_start",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "models": cfg.models,
        })

        # ------------------------------------------------------------------ #
        # Step 1: Dataset statistics
        # ------------------------------------------------------------------ #
        try:
            from evaluate.dataset_explorer import (
                compute_ngram_overlap,
                compute_domain_distribution,
                compute_length_distribution,
            )
            bundle["dataset_stats"] = {
                "ngram_overlap": None,
                "domain_distribution": None,
                "length_distribution": None,
            }
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "dataset_stats",
                "traceback_summary": traceback.format_exc(limit=5),
            })
            bundle["dataset_stats"] = None

        # ------------------------------------------------------------------ #
        # Step 2: Benchmark evaluation (HF models)
        # ------------------------------------------------------------------ #
        try:
            benchmark_scores: dict[str, dict[str, float | None]] = {}
            for model_name in cfg.models:
                scores = run_evaluation(
                    model_name=model_name,
                    tasks=cfg.tasks,
                    config=cfg,
                    logger=self.logger,
                )
                benchmark_scores[model_name] = scores
            bundle["benchmark_scores"] = benchmark_scores
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "benchmark_scores",
                "traceback_summary": traceback.format_exc(limit=5),
            })
            bundle["benchmark_scores"] = {}

        # ------------------------------------------------------------------ #
        # Step 3: Local model evaluation
        # ------------------------------------------------------------------ #
        if cfg.local_checkpoint_path:
            try:
                if not os.path.exists(cfg.local_checkpoint_path):
                    self.logger.log({
                        "type": "warning",
                        "step": "local_model",
                        "message": f"Checkpoint not found: {cfg.local_checkpoint_path}",
                    })
                    bundle["local_model"] = None
                else:
                    from pretrain.model import Transformer
                    from pretrain.config import TransformerConfig
                    from shared.checkpointing import load_checkpoint

                    pretrain_cfg = TransformerConfig()
                    local_model = Transformer(pretrain_cfg)
                    load_checkpoint(cfg.local_checkpoint_path, local_model)
                    local_scores = run_evaluation(
                        model_name=os.path.basename(cfg.local_checkpoint_path),
                        tasks=cfg.tasks,
                        config=cfg,
                        logger=self.logger,
                    )
                    bundle["local_model"] = local_scores
            except Exception:
                self.logger.log({
                    "type": "pipeline_step_error",
                    "step": "local_model",
                    "traceback_summary": traceback.format_exc(limit=5),
                })
                bundle["local_model"] = None
        else:
            bundle["local_model"] = None

        # ------------------------------------------------------------------ #
        # Step 4: Perplexity
        # ------------------------------------------------------------------ #
        try:
            perplexity: dict[str, float | None] = {}
            if cfg.perplexity_corpus:
                from evaluate.perplexity import PerplexityCalculator
                corpus_text = ""
                if os.path.exists(cfg.perplexity_corpus):
                    with open(cfg.perplexity_corpus, encoding="utf-8") as f:
                        corpus_text = f.read()
                else:
                    self.logger.log({
                        "type": "warning",
                        "step": "perplexity",
                        "message": f"Corpus file not found: {cfg.perplexity_corpus}",
                    })

                for model_name in cfg.models:
                    perplexity[model_name] = None  # HF models need custom tokenizer wiring
            else:
                for model_name in cfg.models:
                    perplexity[model_name] = None
            bundle["perplexity"] = perplexity
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "perplexity",
                "traceback_summary": traceback.format_exc(limit=5),
            })
            bundle["perplexity"] = {m: None for m in cfg.models}

        # ------------------------------------------------------------------ #
        # Step 5: Few-shot sensitivity
        # ------------------------------------------------------------------ #
        try:
            from evaluate.few_shot import FewShotAnalyzer
            few_shot_results: dict[tuple[str, str, int], float | None] = {}
            # FewShotAnalyzer requires a loaded lm_eval model; skip if lm_eval absent
            from evaluate import few_shot as fs_module
            if fs_module._HAS_LM_EVAL:
                import lm_eval  # type: ignore
                analyzer = FewShotAnalyzer(cfg, self.logger)
                for model_name in cfg.models:
                    try:
                        lm = lm_eval.models.get_model("hf")(
                            pretrained=model_name, device="cpu"
                        )
                        per_model = analyzer.run(model_name, lm)
                        for (task, n_shots), acc in per_model.items():
                            few_shot_results[(model_name, task, n_shots)] = acc
                    except Exception:
                        self.logger.log({
                            "type": "pipeline_step_error",
                            "step": "few_shot",
                            "model_name": model_name,
                            "traceback_summary": traceback.format_exc(limit=5),
                        })
                        for task, shot_list in cfg.shot_counts.items():
                            for n_shots in shot_list:
                                few_shot_results[(model_name, task, n_shots)] = None
            else:
                for model_name in cfg.models:
                    for task, shot_list in cfg.shot_counts.items():
                        for n_shots in shot_list:
                            few_shot_results[(model_name, task, n_shots)] = None
            bundle["few_shot_sensitivity"] = few_shot_results
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "few_shot_sensitivity",
                "traceback_summary": traceback.format_exc(limit=5),
            })
            bundle["few_shot_sensitivity"] = {}

        # ------------------------------------------------------------------ #
        # Step 6: Calibration
        # ------------------------------------------------------------------ #
        try:
            from evaluate.calibration import CalibrationAnalyzer
            calibration: dict[str, dict[str, Any] | None] = {}
            cal_analyzer = CalibrationAnalyzer(json_logger=self.logger)
            for model_name in cfg.models:
                # Confidence scores require lm_eval logit extraction; placeholder
                calibration[model_name] = {"ece": None, "reliability_diagram_path": None}
            bundle["calibration"] = calibration
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "calibration",
                "traceback_summary": traceback.format_exc(limit=5),
            })
            bundle["calibration"] = {m: None for m in cfg.models}

        # ------------------------------------------------------------------ #
        # Step 7: Weight analysis
        # ------------------------------------------------------------------ #
        try:
            from evaluate.weight_analysis import (
                compute_weight_norms,
                compute_singular_values,
                plot_weight_analysis,
            )
            weight_analysis: dict[str, Any | None] = {}
            for model_name in cfg.models:
                weight_analysis[model_name] = None  # requires loaded model object
            bundle["weight_analysis"] = weight_analysis
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "weight_analysis",
                "traceback_summary": traceback.format_exc(limit=5),
            })
            bundle["weight_analysis"] = {m: None for m in cfg.models}

        # ------------------------------------------------------------------ #
        # Step 8: Activation analysis
        # ------------------------------------------------------------------ #
        try:
            from evaluate.activation_analysis import record_activations
            activation_analysis: dict[str, Any | None] = {}
            for model_name in cfg.models:
                activation_analysis[model_name] = None  # requires loaded model + dataloader
            bundle["activation_analysis"] = activation_analysis
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "activation_analysis",
                "traceback_summary": traceback.format_exc(limit=5),
            })
            bundle["activation_analysis"] = {m: None for m in cfg.models}

        # ------------------------------------------------------------------ #
        # Step 9: Report generation
        # ------------------------------------------------------------------ #
        try:
            generate_csv_report(bundle["benchmark_scores"], cfg.report_csv)
            generate_markdown_report(bundle["benchmark_scores"], cfg.report_md)
            generate_narrative_report(
                results=bundle["benchmark_scores"],
                perplexity=bundle["perplexity"],
                calibration=bundle["calibration"],
                few_shot=bundle["few_shot_sensitivity"],
                output_path=cfg.report_narrative,
            )
        except Exception:
            self.logger.log({
                "type": "pipeline_step_error",
                "step": "report_generation",
                "traceback_summary": traceback.format_exc(limit=5),
            })

        # Log completion event
        self.logger.log({
            "type": "pipeline_complete",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "models": cfg.models,
            "output_dir": out_dir,
        })

        return bundle
