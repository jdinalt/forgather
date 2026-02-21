"""Comprehensive unit tests for the forgather ML analysis module.

Tests cover:
- log_parser.py: TrainingLog dataclass and find_log_files function
- metrics.py: compute_summary_statistics and format_summary_* functions
- plotting.py: smooth_values, plot_training_metrics, plot_loss_curves
"""

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

from forgather.ml.analysis.log_parser import TrainingLog, find_log_files
from forgather.ml.analysis.metrics import (
    compute_summary_statistics,
    format_summary_markdown,
    format_summary_oneline,
    format_summary_text,
)
from forgather.ml.analysis.plotting import (
    plot_loss_curves,
    plot_training_metrics,
    smooth_values,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "timestamp": 1000,
        "global_step": 10,
        "epoch": 0.1,
        "loss": 3.5,
        "grad_norm": 1.0,
        "learning_rate": 0.0001,
    },
    {
        "timestamp": 1100,
        "global_step": 20,
        "epoch": 0.2,
        "loss": 3.0,
        "grad_norm": 0.9,
        "learning_rate": 0.0002,
    },
    {
        "timestamp": 1200,
        "global_step": 30,
        "epoch": 0.3,
        "loss": 2.5,
        "grad_norm": 0.8,
        "learning_rate": 0.0002,
    },
    {
        "eval_loss": 2.8,
        "global_step": 30,
        "epoch": 0.3,
        "timestamp": 1250,
    },
    {
        "timestamp": 1300,
        "global_step": 40,
        "epoch": 0.4,
        "loss": 2.0,
        "grad_norm": 0.7,
        "learning_rate": 0.0001,
    },
    {
        "train_runtime": 300.0,
        "train_samples": 1000,
        "train_samples_per_second": 3.33,
        "train_steps_per_second": 0.13,
        "effective_batch_size": 32,
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_log_file(tmp_path):
    """Create a temporary training log file with sample data."""
    log_file = tmp_path / "trainer_logs.json"
    log_file.write_text(json.dumps(SAMPLE_RECORDS))
    return log_file


@pytest.fixture
def sample_run_dir(tmp_path):
    """Create a temporary run directory with a training log."""
    run_dir = tmp_path / "output_models" / "my_model" / "runs" / "my_run"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "trainer_logs.json"
    log_file.write_text(json.dumps(SAMPLE_RECORDS))
    return run_dir


@pytest.fixture
def sample_log(sample_log_file):
    """Return a TrainingLog loaded from the sample log file."""
    return TrainingLog.from_file(sample_log_file)


@pytest.fixture
def training_only_log(tmp_path):
    """Create a log with training records only (no eval, no final)."""
    records = [
        {
            "timestamp": 1000,
            "global_step": 10,
            "epoch": 0.1,
            "loss": 3.5,
            "grad_norm": 1.0,
            "learning_rate": 0.0001,
        },
        {
            "timestamp": 1100,
            "global_step": 20,
            "epoch": 0.2,
            "loss": 2.5,
            "grad_norm": 0.8,
            "learning_rate": 0.0002,
        },
    ]
    log_file = tmp_path / "train_only.json"
    log_file.write_text(json.dumps(records))
    return TrainingLog.from_file(log_file)


@pytest.fixture
def empty_log(tmp_path):
    """Create a log with an empty records list."""
    log_file = tmp_path / "empty.json"
    log_file.write_text(json.dumps([]))
    return TrainingLog.from_file(log_file)


@pytest.fixture
def multi_eval_log(tmp_path):
    """Create a log with multiple evaluation records."""
    records = [
        {"timestamp": 1000, "global_step": 10, "epoch": 0.1, "loss": 3.5, "grad_norm": 1.0, "learning_rate": 0.0001},
        {"eval_loss": 3.2, "global_step": 10, "epoch": 0.1, "timestamp": 1050},
        {"timestamp": 1100, "global_step": 20, "epoch": 0.2, "loss": 2.5, "grad_norm": 0.8, "learning_rate": 0.0002},
        {"eval_loss": 2.6, "global_step": 20, "epoch": 0.2, "timestamp": 1150},
        {"timestamp": 1200, "global_step": 30, "epoch": 0.3, "loss": 2.0, "grad_norm": 0.7, "learning_rate": 0.0001},
        {"eval_loss": 2.1, "global_step": 30, "epoch": 0.3, "timestamp": 1250},
        {
            "train_runtime": 250.0,
            "train_samples": 800,
            "train_samples_per_second": 3.2,
            "train_steps_per_second": 0.12,
            "effective_batch_size": 16,
        },
    ]
    log_file = tmp_path / "multi_eval.json"
    log_file.write_text(json.dumps(records))
    return TrainingLog.from_file(log_file)


# ===========================================================================
# Tests for log_parser.py
# ===========================================================================


class TestTrainingLogFromFile:
    """Tests for TrainingLog.from_file()."""

    def test_load_valid_file(self, sample_log_file):
        log = TrainingLog.from_file(sample_log_file)
        assert len(log.records) == len(SAMPLE_RECORDS)
        assert log.log_path == sample_log_file

    def test_load_from_string_path(self, sample_log_file):
        log = TrainingLog.from_file(str(sample_log_file))
        assert len(log.records) == len(SAMPLE_RECORDS)
        assert log.log_path == sample_log_file

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Log file not found"):
            TrainingLog.from_file(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            TrainingLog.from_file(bad_file)

    def test_non_array_json(self, tmp_path):
        obj_file = tmp_path / "object.json"
        obj_file.write_text(json.dumps({"key": "value"}))
        with pytest.raises(ValueError, match="must contain a JSON array"):
            TrainingLog.from_file(obj_file)

    def test_empty_array(self, tmp_path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]")
        log = TrainingLog.from_file(empty_file)
        assert log.records == []

    def test_single_record(self, tmp_path):
        single_file = tmp_path / "single.json"
        single_file.write_text(json.dumps([{"loss": 1.0, "global_step": 1}]))
        log = TrainingLog.from_file(single_file)
        assert len(log.records) == 1


class TestTrainingLogFromRunDir:
    """Tests for TrainingLog.from_run_dir()."""

    def test_load_from_run_dir(self, sample_run_dir):
        log = TrainingLog.from_run_dir(sample_run_dir)
        assert len(log.records) == len(SAMPLE_RECORDS)

    def test_missing_log_in_run_dir(self, tmp_path):
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            TrainingLog.from_run_dir(empty_dir)

    def test_from_string_run_dir(self, sample_run_dir):
        log = TrainingLog.from_run_dir(str(sample_run_dir))
        assert len(log.records) == len(SAMPLE_RECORDS)


class TestRunNameExtraction:
    """Tests for __post_init__ run_name extraction."""

    def test_extract_run_name_from_path(self, tmp_path):
        run_dir = tmp_path / "output_models" / "model1" / "runs" / "experiment_v2"
        run_dir.mkdir(parents=True)
        log_file = run_dir / "trainer_logs.json"
        log_file.write_text(json.dumps([]))
        log = TrainingLog.from_file(log_file)
        assert log.run_name == "experiment_v2"

    def test_run_name_none_when_no_runs_in_path(self, tmp_path):
        log_file = tmp_path / "trainer_logs.json"
        log_file.write_text(json.dumps([]))
        log = TrainingLog.from_file(log_file)
        assert log.run_name is None

    def test_run_name_explicit_overrides_path(self, tmp_path):
        run_dir = tmp_path / "output_models" / "model1" / "runs" / "auto_name"
        run_dir.mkdir(parents=True)
        log_file = run_dir / "trainer_logs.json"
        log_file.write_text(json.dumps([]))
        log = TrainingLog(log_path=log_file, records=[], run_name="manual_name")
        assert log.run_name == "manual_name"

    def test_run_name_when_runs_is_last_part(self, tmp_path):
        """If 'runs' is the last directory, there's no segment after it for run_name."""
        run_dir = tmp_path / "runs"
        run_dir.mkdir(parents=True)
        log_file = run_dir / "trainer_logs.json"
        log_file.write_text(json.dumps([]))
        log = TrainingLog.from_file(log_file)
        # "runs" is a dir, the next part is "trainer_logs.json" (the filename)
        # Since parts includes the filename, runs_idx+1 is "trainer_logs.json"
        assert log.run_name == "trainer_logs.json"


class TestGetTrainingRecords:
    """Tests for TrainingLog.get_training_records()."""

    def test_filters_training_records(self, sample_log):
        train_records = sample_log.get_training_records()
        assert len(train_records) == 4
        for r in train_records:
            assert "loss" in r
            assert "eval_loss" not in r

    def test_excludes_eval_records(self, sample_log):
        train_records = sample_log.get_training_records()
        eval_steps = {r["global_step"] for r in sample_log.get_eval_records()}
        for r in train_records:
            # A training record with the same step as an eval record is fine
            # as long as it does NOT have eval_loss itself
            assert "eval_loss" not in r

    def test_excludes_final_summary(self, sample_log):
        train_records = sample_log.get_training_records()
        for r in train_records:
            assert "train_runtime" not in r

    def test_empty_log(self, empty_log):
        assert empty_log.get_training_records() == []


class TestGetEvalRecords:
    """Tests for TrainingLog.get_eval_records()."""

    def test_filters_eval_records(self, sample_log):
        eval_records = sample_log.get_eval_records()
        assert len(eval_records) == 1
        assert eval_records[0]["eval_loss"] == 2.8
        assert eval_records[0]["global_step"] == 30

    def test_multiple_eval_records(self, multi_eval_log):
        eval_records = multi_eval_log.get_eval_records()
        assert len(eval_records) == 3
        assert [r["eval_loss"] for r in eval_records] == [3.2, 2.6, 2.1]

    def test_no_eval_records(self, training_only_log):
        assert training_only_log.get_eval_records() == []

    def test_empty_log(self, empty_log):
        assert empty_log.get_eval_records() == []


class TestGetFinalRecord:
    """Tests for TrainingLog.get_final_record()."""

    def test_finds_final_record(self, sample_log):
        final = sample_log.get_final_record()
        assert final is not None
        assert final["train_runtime"] == 300.0
        assert final["train_samples"] == 1000
        assert final["effective_batch_size"] == 32

    def test_no_final_record(self, training_only_log):
        assert training_only_log.get_final_record() is None

    def test_empty_log(self, empty_log):
        assert empty_log.get_final_record() is None

    def test_final_record_when_not_last(self, tmp_path):
        """Final record detection should work even if train_runtime isn't the very last record."""
        records = [
            {"loss": 2.0, "global_step": 10},
            {"train_runtime": 100.0, "train_samples": 500},
            {"loss": 1.5, "global_step": 20},  # Some record after
        ]
        log_file = tmp_path / "out_of_order.json"
        log_file.write_text(json.dumps(records))
        log = TrainingLog.from_file(log_file)
        # get_final_record iterates in reverse, so it still finds the train_runtime record
        final = log.get_final_record()
        assert final is not None
        assert final["train_runtime"] == 100.0


class TestGetMetricValues:
    """Tests for TrainingLog.get_metric_values()."""

    def test_extract_loss_from_all_records(self, sample_log):
        losses = sample_log.get_metric_values("loss")
        assert losses == [3.5, 3.0, 2.5, 2.0]

    def test_extract_from_specific_records(self, sample_log):
        train_records = sample_log.get_training_records()
        lr_values = sample_log.get_metric_values("learning_rate", train_records)
        assert lr_values == [0.0001, 0.0002, 0.0002, 0.0001]

    def test_metric_not_present(self, sample_log):
        result = sample_log.get_metric_values("nonexistent_metric")
        assert result == []

    def test_metric_in_some_records(self, sample_log):
        eval_losses = sample_log.get_metric_values("eval_loss")
        assert eval_losses == [2.8]

    def test_empty_log(self, empty_log):
        assert empty_log.get_metric_values("loss") == []


class TestGetSteps:
    """Tests for TrainingLog.get_steps()."""

    def test_get_all_steps(self, sample_log):
        steps = sample_log.get_steps()
        assert steps == [10, 20, 30, 30, 40]

    def test_get_steps_from_training_records(self, sample_log):
        train_records = sample_log.get_training_records()
        steps = sample_log.get_steps(train_records)
        assert steps == [10, 20, 30, 40]

    def test_empty_log(self, empty_log):
        assert empty_log.get_steps() == []


class TestGetEpochs:
    """Tests for TrainingLog.get_epochs()."""

    def test_get_all_epochs(self, sample_log):
        epochs = sample_log.get_epochs()
        assert epochs == [0.1, 0.2, 0.3, 0.3, 0.4]

    def test_get_epochs_from_eval_records(self, sample_log):
        eval_records = sample_log.get_eval_records()
        epochs = sample_log.get_epochs(eval_records)
        assert epochs == [0.3]


class TestGetTimestamps:
    """Tests for TrainingLog.get_timestamps()."""

    def test_get_all_timestamps(self, sample_log):
        timestamps = sample_log.get_timestamps()
        assert timestamps == [1000, 1100, 1200, 1250, 1300]

    def test_get_timestamps_from_training_records(self, sample_log):
        train_records = sample_log.get_training_records()
        timestamps = sample_log.get_timestamps(train_records)
        assert timestamps == [1000, 1100, 1200, 1300]


class TestFindBestStep:
    """Tests for TrainingLog.find_best_step()."""

    def test_find_min_loss(self, sample_log):
        result = sample_log.find_best_step("loss", mode="min")
        assert result is not None
        step, value = result
        assert step == 40
        assert value == 2.0

    def test_find_max_loss(self, sample_log):
        result = sample_log.find_best_step("loss", mode="max")
        assert result is not None
        step, value = result
        assert step == 10
        assert value == 3.5

    def test_find_min_eval_loss(self, multi_eval_log):
        result = multi_eval_log.find_best_step("eval_loss", mode="min")
        assert result is not None
        step, value = result
        assert step == 30
        assert value == 2.1

    def test_find_max_learning_rate(self, sample_log):
        result = sample_log.find_best_step("learning_rate", mode="max")
        assert result is not None
        step, value = result
        assert value == 0.0002

    def test_metric_not_found(self, sample_log):
        result = sample_log.find_best_step("nonexistent_metric")
        assert result is None

    def test_empty_log(self, empty_log):
        result = empty_log.find_best_step("loss")
        assert result is None

    def test_find_min_grad_norm(self, sample_log):
        result = sample_log.find_best_step("grad_norm", mode="min")
        assert result is not None
        step, value = result
        assert step == 40
        assert value == 0.7


class TestFindLogFiles:
    """Tests for find_log_files()."""

    def test_find_logs_in_project(self, tmp_path):
        # Create project structure with multiple runs
        for model in ["model_a", "model_b"]:
            for run in ["run1", "run2"]:
                run_dir = tmp_path / "output_models" / model / "runs" / run
                run_dir.mkdir(parents=True)
                log_file = run_dir / "trainer_logs.json"
                log_file.write_text(json.dumps([]))
                # Ensure different mtime for deterministic ordering
                time.sleep(0.05)

        log_files = find_log_files(tmp_path)
        assert len(log_files) == 4
        # Should be sorted by mtime, most recent first
        for lf in log_files:
            assert lf.name == "trainer_logs.json"

    def test_find_logs_by_model_name(self, tmp_path):
        for model in ["model_a", "model_b"]:
            run_dir = tmp_path / "output_models" / model / "runs" / "run1"
            run_dir.mkdir(parents=True)
            (run_dir / "trainer_logs.json").write_text(json.dumps([]))

        log_files = find_log_files(tmp_path, model_name="model_a")
        assert len(log_files) == 1
        assert "model_a" in str(log_files[0])

    def test_no_output_models_dir(self, tmp_path):
        result = find_log_files(tmp_path)
        assert result == []

    def test_empty_output_models_dir(self, tmp_path):
        (tmp_path / "output_models").mkdir()
        result = find_log_files(tmp_path)
        assert result == []

    def test_sorted_by_mtime_descending(self, tmp_path):
        run1_dir = tmp_path / "output_models" / "model" / "runs" / "old_run"
        run1_dir.mkdir(parents=True)
        (run1_dir / "trainer_logs.json").write_text(json.dumps([]))

        time.sleep(0.1)

        run2_dir = tmp_path / "output_models" / "model" / "runs" / "new_run"
        run2_dir.mkdir(parents=True)
        (run2_dir / "trainer_logs.json").write_text(json.dumps([]))

        log_files = find_log_files(tmp_path)
        assert len(log_files) == 2
        # Most recent first
        assert "new_run" in str(log_files[0])
        assert "old_run" in str(log_files[1])


# ===========================================================================
# Tests for metrics.py
# ===========================================================================


class TestComputeSummaryStatistics:
    """Tests for compute_summary_statistics()."""

    def test_full_summary(self, sample_log):
        summary = compute_summary_statistics(sample_log)

        # Basic info
        assert "run_name" in summary
        assert "log_path" in summary

        # Training progress
        assert summary["total_steps"] == 40
        assert summary["final_epoch"] == 0.4

        # Loss metrics
        assert summary["final_loss"] == 2.0
        assert summary["min_loss"] == 2.0
        assert summary["best_loss"] == 2.0
        assert summary["best_loss_step"] == 40
        assert abs(summary["avg_loss"] - (3.5 + 3.0 + 2.5 + 2.0) / 4) < 1e-6

        # Gradient statistics
        assert abs(summary["avg_grad_norm"] - (1.0 + 0.9 + 0.8 + 0.7) / 4) < 1e-6
        assert summary["max_grad_norm_value"] == 1.0
        assert summary["max_grad_norm_step"] == 10

        # Learning rate
        assert summary["initial_lr"] == 0.0001
        assert summary["final_lr"] == 0.0001

        # Eval metrics
        assert summary["final_eval_loss"] == 2.8
        assert summary["best_eval_loss"] == 2.8
        assert summary["best_eval_loss_step"] == 30

        # Performance
        assert summary["train_runtime"] == 300.0
        assert summary["train_samples"] == 1000
        assert summary["train_samples_per_second"] == 3.33
        assert summary["train_steps_per_second"] == 0.13
        assert summary["effective_batch_size"] == 32

    def test_summary_without_eval(self, training_only_log):
        summary = compute_summary_statistics(training_only_log)

        # Should have training metrics
        assert summary["total_steps"] == 20
        assert summary["final_loss"] == 2.5
        assert summary["best_loss"] == 2.5

        # Should NOT have eval metrics
        assert "final_eval_loss" not in summary
        assert "best_eval_loss" not in summary

        # Should NOT have performance (no final record)
        assert "train_runtime" not in summary

    def test_summary_empty_log(self, empty_log):
        summary = compute_summary_statistics(empty_log)

        # Should only have basic info
        assert "run_name" in summary
        assert "log_path" in summary

        # Should not have any training metrics
        assert "total_steps" not in summary
        assert "final_loss" not in summary
        assert "final_eval_loss" not in summary
        assert "train_runtime" not in summary

    def test_summary_with_multiple_evals(self, multi_eval_log):
        summary = compute_summary_statistics(multi_eval_log)

        assert summary["final_eval_loss"] == 2.1
        assert summary["best_eval_loss"] == 2.1
        assert summary["best_eval_loss_step"] == 30

    def test_summary_run_name_from_path(self, sample_run_dir):
        log = TrainingLog.from_run_dir(sample_run_dir)
        summary = compute_summary_statistics(log)
        assert summary["run_name"] == "my_run"


class TestFormatSummaryText:
    """Tests for format_summary_text()."""

    def test_basic_output(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)

        assert "Training Run Summary" in text
        assert "=" * 60 in text

    def test_contains_run_name(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        summary["run_name"] = "test_experiment"
        text = format_summary_text(summary)
        assert "test_experiment" in text

    def test_contains_duration(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)
        assert "Duration: 300.00s" in text

    def test_contains_total_steps(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)
        assert "Total Steps: 40" in text

    def test_contains_loss_metrics(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)
        assert "Metrics:" in text
        assert "Final Loss:" in text
        assert "Best Loss:" in text
        assert "Average Loss:" in text

    def test_contains_eval_metrics(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)
        assert "Final Eval Loss:" in text
        assert "Best Eval Loss:" in text

    def test_contains_training_speed(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)
        assert "Training Speed:" in text
        assert "Samples/sec:" in text
        assert "Steps/sec:" in text
        assert "Effective Batch Size:" in text

    def test_contains_gradient_statistics(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)
        assert "Gradient Statistics:" in text
        assert "Average Grad Norm:" in text
        assert "Max Grad Norm:" in text

    def test_contains_learning_rate(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        text = format_summary_text(summary)
        assert "Learning Rate:" in text
        assert "Initial:" in text
        assert "Final:" in text

    def test_missing_run_name_shows_unknown(self):
        summary = {}
        text = format_summary_text(summary)
        assert "Unknown" in text

    def test_none_run_name_shows_none(self):
        # When run_name is explicitly None (not absent), .get() returns None
        # and the text shows "None" as the run name.
        summary = {"run_name": None}
        text = format_summary_text(summary)
        assert "None" in text

    def test_minimal_summary(self):
        summary = {"run_name": "test", "log_path": "/tmp/test.json"}
        text = format_summary_text(summary)
        assert "Training Run Summary" in text
        assert "test" in text


class TestFormatSummaryMarkdown:
    """Tests for format_summary_markdown()."""

    def test_markdown_header(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "# Training Run Summary" in md

    def test_markdown_table_headers(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "| Metric | Value | Step |" in md
        assert "|--------|-------|------|" in md

    def test_markdown_metrics_section(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "## Metrics" in md

    def test_markdown_speed_section(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "## Training Speed" in md

    def test_markdown_bold_formatting(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "**Run:**" in md

    def test_markdown_contains_loss_values(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "Final Loss" in md
        assert "Best Loss" in md

    def test_markdown_contains_eval_loss(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "Final Eval Loss" in md
        assert "Best Eval Loss" in md

    def test_markdown_no_eval_section_when_missing(self, training_only_log):
        summary = compute_summary_statistics(training_only_log)
        md = format_summary_markdown(summary)
        assert "Eval Loss" not in md

    def test_markdown_contains_samples_per_sec(self, sample_log):
        summary = compute_summary_statistics(sample_log)
        md = format_summary_markdown(summary)
        assert "Samples/sec" in md

    def test_minimal_summary(self):
        summary = {"run_name": "test", "log_path": "/tmp/test.json"}
        md = format_summary_markdown(summary)
        assert "# Training Run Summary" in md


class TestFormatSummaryOneline:
    """Tests for format_summary_oneline()."""

    def _make_summary(self, log, run_name="test_run"):
        """Compute summary and set a non-None run_name to avoid the None truncation bug."""
        summary = compute_summary_statistics(log)
        summary["run_name"] = run_name
        return summary

    def test_basic_format(self, sample_log):
        summary = self._make_summary(sample_log)
        line = format_summary_oneline(summary)
        assert "|" in line  # Pipe separator

    def test_contains_steps(self, sample_log):
        summary = self._make_summary(sample_log)
        line = format_summary_oneline(summary)
        assert "steps=40" in line

    def test_contains_duration(self, sample_log):
        summary = self._make_summary(sample_log)
        line = format_summary_oneline(summary)
        assert "time=05:00" in line  # 300 seconds = 5 minutes

    def test_contains_loss(self, sample_log):
        summary = self._make_summary(sample_log)
        line = format_summary_oneline(summary)
        assert "loss=2.0000" in line

    def test_contains_eval(self, sample_log):
        summary = self._make_summary(sample_log)
        line = format_summary_oneline(summary)
        assert "eval=2.8000" in line

    def test_contains_samples_per_second(self, sample_log):
        summary = self._make_summary(sample_log)
        line = format_summary_oneline(summary)
        assert "samp/s=3.3" in line

    def test_no_eval_when_missing(self, training_only_log):
        summary = self._make_summary(training_only_log)
        line = format_summary_oneline(summary)
        assert "eval=" not in line

    def test_duration_na_when_missing(self, training_only_log):
        summary = self._make_summary(training_only_log)
        line = format_summary_oneline(summary)
        assert "time=N/A" in line

    def test_run_name_truncation(self):
        summary = {
            "run_name": "a_very_long_run_name_that_exceeds_thirty_characters_limit",
            "total_steps": 100,
        }
        line = format_summary_oneline(summary)
        # Run name should be truncated to 30 chars
        assert "a_very_long_run_name_that_exce" in line

    def test_missing_run_name_uses_unknown(self):
        """When run_name key is absent, 'Unknown' is used."""
        summary = {"total_steps": 0}
        line = format_summary_oneline(summary)
        assert "Unknown" in line
        assert "steps=0" in line

    def test_none_run_name_raises_type_error(self, sample_log):
        """Known bug: when run_name is None, [:30] fails with TypeError."""
        summary = compute_summary_statistics(sample_log)
        assert summary["run_name"] is None
        with pytest.raises(TypeError):
            format_summary_oneline(summary)

    def test_with_named_run(self, sample_run_dir):
        """With a proper run directory, run_name is set and oneline works."""
        log = TrainingLog.from_run_dir(sample_run_dir)
        summary = compute_summary_statistics(log)
        assert summary["run_name"] == "my_run"
        line = format_summary_oneline(summary)
        assert "my_run" in line


# ===========================================================================
# Tests for plotting.py
# ===========================================================================


class TestSmoothValues:
    """Tests for smooth_values()."""

    def test_no_smoothing_window_1(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = smooth_values(values, window_size=1)
        assert result == values

    def test_window_larger_than_list(self):
        values = [1.0, 2.0, 3.0]
        result = smooth_values(values, window_size=10)
        assert result == values

    def test_window_equal_to_list(self):
        values = [1.0, 2.0, 3.0]
        result = smooth_values(values, window_size=3)
        # Should NOT return the original (3 is not > 3)
        # window_size <= 1 or len(values) < window_size -> return values
        # 3 <= 1 is False, 3 < 3 is False, so smoothing is applied
        assert len(result) == 3
        # Each element is the average of its window
        # i=0: start=0, end=min(3,0+1+1)=2 -> avg(1,2) = 1.5
        # i=1: start=max(0,1-1)=0, end=min(3,1+1+1)=3 -> avg(1,2,3) = 2.0
        # i=2: start=max(0,2-1)=1, end=min(3,2+1+1)=3 -> avg(2,3) = 2.5
        assert abs(result[0] - 1.5) < 1e-6
        assert abs(result[1] - 2.0) < 1e-6
        assert abs(result[2] - 2.5) < 1e-6

    def test_basic_smoothing(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = smooth_values(values, window_size=3)
        assert len(result) == 5
        # i=0: start=0, end=min(5,0+2)=2 -> avg(10,20)=15.0
        # i=1: start=0, end=min(5,1+2)=3 -> avg(10,20,30)=20.0
        # i=2: start=1, end=min(5,2+2)=4 -> avg(20,30,40)=30.0
        # i=3: start=2, end=min(5,3+2)=5 -> avg(30,40,50)=40.0
        # i=4: start=3, end=min(5,4+2)=5 -> avg(40,50)=45.0
        assert abs(result[0] - 15.0) < 1e-6
        assert abs(result[1] - 20.0) < 1e-6
        assert abs(result[2] - 30.0) < 1e-6
        assert abs(result[3] - 40.0) < 1e-6
        assert abs(result[4] - 45.0) < 1e-6

    def test_constant_values(self):
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = smooth_values(values, window_size=3)
        for v in result:
            assert abs(v - 5.0) < 1e-6

    def test_single_value(self):
        values = [42.0]
        result = smooth_values(values, window_size=5)
        assert result == [42.0]

    def test_empty_list(self):
        result = smooth_values([], window_size=3)
        assert result == []

    def test_window_size_zero(self):
        values = [1.0, 2.0, 3.0]
        result = smooth_values(values, window_size=0)
        assert result == values

    def test_preserves_length(self):
        values = list(range(100))
        result = smooth_values([float(v) for v in values], window_size=10)
        assert len(result) == len(values)

    def test_large_window(self):
        values = [1.0, 2.0]
        result = smooth_values(values, window_size=100)
        assert result == values


class TestPlotTrainingMetrics:
    """Tests for plot_training_metrics()."""

    def test_returns_figure(self, sample_log):
        fig = plot_training_metrics([sample_log])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_metric(self, sample_log):
        fig = plot_training_metrics([sample_log], metrics=["loss"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_metrics(self, sample_log):
        fig = plot_training_metrics([sample_log], metrics=["loss", "grad_norm"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_default_metrics(self, sample_log):
        fig = plot_training_metrics([sample_log])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_smoothing(self, sample_log):
        fig = plot_training_metrics([sample_log], smooth_window=2)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_epoch_x_axis(self, sample_log):
        fig = plot_training_metrics([sample_log], metrics=["loss"], x_axis="epoch")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_time_x_axis(self, sample_log):
        fig = plot_training_metrics([sample_log], metrics=["loss"], x_axis="time")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_invalid_x_axis(self, sample_log):
        with pytest.raises(ValueError, match="Invalid x_axis"):
            plot_training_metrics([sample_log], metrics=["loss"], x_axis="invalid")

    def test_log_scale(self, sample_log):
        fig = plot_training_metrics([sample_log], metrics=["loss"], log_scale=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_save_to_file(self, sample_log, tmp_path):
        output_path = str(tmp_path / "test_plot.png")
        fig = plot_training_metrics([sample_log], output_path=output_path)
        assert isinstance(fig, Figure)
        assert os.path.exists(output_path)
        plt.close(fig)

    def test_save_creates_parent_dirs(self, sample_log, tmp_path):
        output_path = str(tmp_path / "subdir" / "nested" / "plot.png")
        fig = plot_training_metrics([sample_log], output_path=output_path)
        assert os.path.exists(output_path)
        plt.close(fig)

    def test_custom_figsize(self, sample_log):
        fig = plot_training_metrics([sample_log], figsize=(8, 4))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_logs(self, sample_log, multi_eval_log):
        fig = plot_training_metrics([sample_log, multi_eval_log], metrics=["loss"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_log_raises_unbound_error(self, empty_log):
        # Known bug: x_label is not initialized before the loop over logs,
        # so when all logs have empty records, ax.set_xlabel(x_label) fails.
        with pytest.raises(UnboundLocalError):
            plot_training_metrics([empty_log], metrics=["loss"])


class TestPlotLossCurves:
    """Tests for plot_loss_curves()."""

    def test_returns_figure(self, sample_log):
        fig = plot_loss_curves([sample_log])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_eval_data(self, multi_eval_log):
        fig = plot_loss_curves([multi_eval_log])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_epoch_x_axis(self, sample_log):
        fig = plot_loss_curves([sample_log], x_axis="epoch")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_time_x_axis(self, sample_log):
        fig = plot_loss_curves([sample_log], x_axis="time")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_invalid_x_axis(self, sample_log):
        with pytest.raises(ValueError, match="Invalid x_axis"):
            plot_loss_curves([sample_log], x_axis="invalid")

    def test_with_smoothing(self, sample_log):
        fig = plot_loss_curves([sample_log], smooth_window=2)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_save_to_file(self, sample_log, tmp_path):
        output_path = str(tmp_path / "loss_curves.png")
        fig = plot_loss_curves([sample_log], output_path=output_path)
        assert isinstance(fig, Figure)
        assert os.path.exists(output_path)
        plt.close(fig)

    def test_save_creates_parent_dirs(self, sample_log, tmp_path):
        output_path = str(tmp_path / "deep" / "dir" / "loss.png")
        fig = plot_loss_curves([sample_log], output_path=output_path)
        assert os.path.exists(output_path)
        plt.close(fig)

    def test_multiple_logs(self, sample_log, multi_eval_log):
        fig = plot_loss_curves([sample_log, multi_eval_log])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_training_only_no_eval(self, training_only_log):
        fig = plot_loss_curves([training_only_log])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_log_raises_unbound_error(self, empty_log):
        # Known bug: x_label is not initialized before the loop over logs,
        # so when all logs have empty records, ax1.set_xlabel(x_label) fails.
        with pytest.raises(UnboundLocalError):
            plot_loss_curves([empty_log])
