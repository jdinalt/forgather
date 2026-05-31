"""Unit tests for the DiLoCo server StatsAggregator."""

import pytest

from forgather.ml.diloco.stats import StatsAggregator


class TestDeltaAccumulation:
    def test_first_report_adds_full_cumulative(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 1000, "flos_total": 5.0, "step_total": 10})
        s = agg.snapshot()
        assert s["total_tokens"] == 1000
        assert s["total_flos"] == 5.0
        assert s["total_steps"] == 10

    def test_subsequent_reports_add_increments(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 1000, "step_total": 10})
        agg.update("w0", {"tokens_total": 2500, "step_total": 25})
        s = agg.snapshot()
        assert s["total_tokens"] == 2500  # 1000 + 1500
        assert s["total_steps"] == 25

    def test_two_workers_sum(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 1000})
        agg.update("w1", {"tokens_total": 400})
        agg.update("w0", {"tokens_total": 1200})
        assert agg.snapshot()["total_tokens"] == 1600  # 1200 + 400

    def test_negative_delta_clamped(self):
        # A worker that resets its counter must not subtract from the total.
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 1000})
        agg.update("w0", {"tokens_total": 200})  # counter reset
        s = agg.snapshot()
        assert s["total_tokens"] == 1000
        # ...and subsequent deltas are measured from the new (lower) baseline.
        agg.update("w0", {"tokens_total": 350})
        assert agg.snapshot()["total_tokens"] == 1150

    def test_reused_worker_id_continues_not_doublecounts(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 5000})
        # Worker dies, dropped from gauges, then a resumed worker reuses the id
        # and reports the same cumulative it had restored from its checkpoint.
        agg.drop_worker("w0")
        agg.update("w0", {"tokens_total": 5000})
        assert agg.snapshot()["total_tokens"] == 5000  # no double count
        agg.update("w0", {"tokens_total": 6000})
        assert agg.snapshot()["total_tokens"] == 6000


class TestLiveGauges:
    def test_throughput_mfu_memory_sum_over_workers(self):
        agg = StatsAggregator()
        agg.update("w0", {"tok_per_sec": 100.0, "mfu": 0.2, "peak_mem": 1e9})
        agg.update("w1", {"tok_per_sec": 150.0, "mfu": 0.3, "peak_mem": 2e9})
        s = agg.snapshot()
        assert s["tok_per_sec"] == 250.0
        assert s["mfu"] == pytest.approx(0.5)
        assert s["peak_memory"] == pytest.approx(3e9)

    def test_latest_value_replaces_previous(self):
        agg = StatsAggregator()
        agg.update("w0", {"tok_per_sec": 100.0})
        agg.update("w0", {"tok_per_sec": 120.0})
        assert agg.snapshot()["tok_per_sec"] == 120.0

    def test_drop_worker_removes_from_gauge(self):
        agg = StatsAggregator()
        agg.update("w0", {"tok_per_sec": 100.0})
        agg.update("w1", {"tok_per_sec": 150.0})
        agg.drop_worker("w0")
        assert agg.snapshot()["tok_per_sec"] == 150.0

    def test_grad_norm_token_weighted_mean(self):
        agg = StatsAggregator()
        agg.update("w0", {"grad_norm": 1.0, "tokens_window": 100})
        agg.update("w1", {"grad_norm": 2.0, "tokens_window": 300})
        # (1*100 + 2*300) / 400 = 1.75
        assert agg.snapshot()["grad_norm"] == pytest.approx(1.75)

    def test_grad_norm_equal_weight_when_no_window(self):
        agg = StatsAggregator()
        agg.update("w0", {"grad_norm": 1.0})
        agg.update("w1", {"grad_norm": 3.0})
        assert agg.snapshot()["grad_norm"] == pytest.approx(2.0)


class TestLossEma:
    def test_train_loss_none_until_reported(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 10})
        assert agg.snapshot()["train_loss"] is None

    def test_constant_loss_converges_to_value(self):
        agg = StatsAggregator(train_loss_decay=0.9)
        for _ in range(50):
            agg.update("w0", {"loss": 3.0, "tokens_window": 100})
        assert agg.snapshot()["train_loss"] == pytest.approx(3.0, abs=1e-6)

    def test_higher_token_report_dominates(self):
        agg = StatsAggregator(train_loss_decay=0.9)
        agg.update("w0", {"loss": 1.0, "tokens_window": 10})
        agg.update("w1", {"loss": 5.0, "tokens_window": 1000})
        # The big-token report pulls the weighted EMA toward 5.
        assert agg.snapshot()["train_loss"] > 4.0

    def test_eval_loss_and_step_tracked(self):
        agg = StatsAggregator(eval_loss_decay=0.5)
        agg.update("w0", {"eval_loss": 2.0, "eval_step": 100, "tokens_window": 100})
        s = agg.snapshot()
        assert s["eval_loss"] == pytest.approx(2.0)
        assert s["eval_step"] == 100

    def test_eval_weaker_smoothing_than_train(self):
        # With a weaker decay, eval tracks a step change faster than train.
        train = StatsAggregator(train_loss_decay=0.9)
        ev = StatsAggregator(eval_loss_decay=0.5)
        for _ in range(3):
            train.update("w", {"loss": 1.0, "tokens_window": 1})
            ev.update("w", {"eval_loss": 1.0, "tokens_window": 1})
        for _ in range(3):
            train.update("w", {"loss": 2.0, "tokens_window": 1})
            ev.update("w", {"eval_loss": 2.0, "tokens_window": 1})
        # Both rising toward 2; the weaker-smoothed eval is closer to it.
        assert ev.snapshot()["eval_loss"] > train.snapshot()["train_loss"]


class TestPersistence:
    def test_roundtrip_preserves_totals_and_ema(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 1000, "loss": 3.0, "tokens_window": 100})
        agg.update("w0", {"eval_loss": 2.5, "eval_step": 50, "tokens_window": 100})
        state = agg.state_dict()

        restored = StatsAggregator()
        restored.load_state_dict(state)
        s0, s1 = agg.snapshot(), restored.snapshot()
        assert s1["total_tokens"] == s0["total_tokens"]
        assert s1["train_loss"] == pytest.approx(s0["train_loss"])
        assert s1["eval_loss"] == pytest.approx(s0["eval_loss"])
        assert s1["eval_step"] == s0["eval_step"]

    def test_resume_does_not_doublecount_same_worker(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 5000})
        state = agg.state_dict()

        # Server restarts from checkpoint; worker resumes with the same id and
        # reports the cumulative it restored from its own checkpoint.
        restored = StatsAggregator()
        restored.load_state_dict(state)
        restored.update("w0", {"tokens_total": 5000})
        assert restored.snapshot()["total_tokens"] == 5000
        restored.update("w0", {"tokens_total": 5500})
        assert restored.snapshot()["total_tokens"] == 5500

    def test_load_empty_state_is_noop(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 123})
        agg.load_state_dict({})
        assert agg.snapshot()["total_tokens"] == 123

    def test_load_tolerates_missing_keys(self):
        agg = StatsAggregator()
        agg.load_state_dict({"total_tokens": 42})
        s = agg.snapshot()
        assert s["total_tokens"] == 42
        assert s["total_flos"] == 0.0
        assert s["train_loss"] is None


class TestRobustness:
    def test_empty_and_missing_inputs(self):
        agg = StatsAggregator()
        agg.update("", {"tokens_total": 10})
        agg.update("w0", {})
        agg.update("w0", None)
        assert agg.snapshot()["total_tokens"] == 0

    def test_partial_snapshot_only_updates_present_fields(self):
        agg = StatsAggregator()
        agg.update("w0", {"tokens_total": 100})
        agg.update("w0", {"loss": 2.0, "tokens_window": 10})
        s = agg.snapshot()
        assert s["total_tokens"] == 100
        assert s["train_loss"] == pytest.approx(2.0)
