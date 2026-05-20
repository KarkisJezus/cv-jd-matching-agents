"""
Phase 6 tests: Evaluation metrics, experiment runner, comparison report.

Tests cover:
1. Metric computation from SharedContext
2. ComparisonReport aggregation
3. ExperimentRunner end-to-end with mock LLM
4. Dataset loading from JSON
5. Report formatting
6. Results persistence
7. Classification metrics (precision, recall, F1)
8. Trace recording and saving
"""

import json
from pathlib import Path

import pytest

from evaluation.metrics import (
    ClassificationMetrics,
    ComparisonReport,
    ScenarioMetrics,
    compute_classification_metrics,
    compute_metrics,
    format_comparison_table,
)
from evaluation.runner import (
    CVJDPair,
    ExperimentConfig,
    ExperimentRunner,
    load_dataset_from_json,
)
from llm.client import MockLLMClient
from orchestrator.orchestrator import Orchestrator


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.fixture
def sample_context(mock_llm):
    """A completed context from a Scenario A run."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    return orchestrator.run("test cv text", "test jd text", scenario="A")


@pytest.fixture
def sample_pair():
    return CVJDPair(
        pair_id="test_pair",
        cv_text="Python developer with ML experience",
        jd_text="Looking for ML engineer with deep learning",
        ground_truth_score=60.0,
        ground_truth_label=True,
        description="Test pair",
    )


@pytest.fixture
def sample_dataset():
    return [
        CVJDPair(
            pair_id="pair_1",
            cv_text="Python developer with data analysis and ML skills",
            jd_text="ML engineer position requiring deep learning and Python",
            ground_truth_score=65.0,
            ground_truth_label=True,
            description="Partial match",
        ),
        CVJDPair(
            pair_id="pair_2",
            cv_text="Java backend developer with Spring Boot",
            jd_text="Senior Java developer for microservices",
            ground_truth_score=80.0,
            ground_truth_label=True,
            description="Good match",
        ),
    ]


# ── Test 1: Metric computation ───────────────────────────────

def test_compute_metrics_from_context(sample_context):
    """compute_metrics produces valid ScenarioMetrics from a completed context."""
    metrics = compute_metrics(sample_context, pair_id="test")

    assert metrics.scenario == "A"
    assert metrics.pair_id == "test"
    assert metrics.final_score > 0
    assert 0 <= metrics.confidence <= 1
    assert metrics.recommendation != ""
    assert metrics.reasoning_suggested_score > 0
    assert metrics.total_agents_executed > 0
    assert metrics.total_log_entries > 0


def test_compute_metrics_score_delta(sample_context):
    """Score delta is the absolute difference between final and suggested scores."""
    metrics = compute_metrics(sample_context)

    expected_delta = abs(
        sample_context.final_decision.score
        - sample_context.reasoning_output.suggested_score
    )
    assert metrics.score_delta == pytest.approx(expected_delta, abs=0.01)


def test_compute_metrics_matching(sample_context):
    """Matching metrics are correctly extracted."""
    metrics = compute_metrics(sample_context)

    assert metrics.similarity_overall > 0
    assert 0 <= metrics.coverage_ratio <= 1
    assert metrics.total_jd_skills > 0


def test_compute_metrics_reflection(sample_context):
    """Reflection metrics are correctly extracted."""
    metrics = compute_metrics(sample_context)

    assert metrics.reflection_triggered is True
    assert 0 <= metrics.reflection_confidence <= 1
    assert metrics.was_revised is True  # Mock triggers one revision


def test_compute_metrics_no_enrichment_for_scenario_a(sample_context):
    """Scenario A should have no enrichment metrics."""
    metrics = compute_metrics(sample_context)

    assert metrics.has_enrichment is False
    assert metrics.taxonomy_mapped_count == 0


def test_compute_metrics_scenario_b_has_enrichment(mock_llm):
    """Scenario B should have enrichment metrics."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    context = orchestrator.run("cv text", "jd text", scenario="B")
    metrics = compute_metrics(context)

    assert metrics.has_enrichment is True
    assert metrics.cv_skills_normalized > 0
    assert metrics.jd_skills_normalized > 0
    assert metrics.taxonomy_mapped_count > 0


# ── Test 2: ComparisonReport ─────────────────────────────────

def test_comparison_report_add_pair():
    """ComparisonReport correctly stores pair results."""
    report = ComparisonReport()

    m_a = ScenarioMetrics(scenario="A", final_score=70, confidence=0.8)
    m_b = ScenarioMetrics(scenario="B", final_score=72, confidence=0.82)

    report.add_pair("pair_1", metrics_a=m_a, metrics_b=m_b, ground_truth_score=75)

    assert len(report.pair_results) == 1
    pair = report.pair_results[0]
    assert pair["pair_id"] == "pair_1"
    assert pair["score_A"] == 70
    assert pair["score_B"] == 72
    assert pair["error_A"] == 5.0
    assert pair["error_B"] == 3.0


def test_comparison_report_summary():
    """compute_summary produces aggregate statistics."""
    report = ComparisonReport()

    report.add_pair(
        "pair_1",
        metrics_a=ScenarioMetrics(scenario="A", final_score=70, confidence=0.8),
        metrics_b=ScenarioMetrics(scenario="B", final_score=72, confidence=0.85),
    )
    report.add_pair(
        "pair_2",
        metrics_a=ScenarioMetrics(scenario="A", final_score=60, confidence=0.7),
        metrics_b=ScenarioMetrics(scenario="B", final_score=65, confidence=0.75),
    )

    summary = report.compute_summary()

    assert "A" in summary
    assert "B" in summary
    assert summary["A"]["n_pairs"] == 2
    assert summary["A"]["mean_score"] == pytest.approx(65.0)
    assert summary["B"]["mean_score"] == pytest.approx(68.5)


# ── Test 3: ExperimentRunner ─────────────────────────────────

def test_experiment_runner_runs_all_scenarios(mock_llm, sample_dataset, tmp_path):
    """ExperimentRunner executes all scenarios on all pairs."""
    config = ExperimentConfig(
        scenarios=["A", "B"],  # Skip C to keep test fast
        output_dir=str(tmp_path),
        experiment_name="test_exp",
    )

    runner = ExperimentRunner(mock_llm, config)
    report = runner.run_all(sample_dataset)

    # Should have results for both pairs
    assert len(report.pair_results) == 2

    # Each pair should have both scenario scores
    for pair in report.pair_results:
        assert "score_A" in pair
        assert "score_B" in pair

    # Summary should be computed
    assert "A" in report.summary
    assert "B" in report.summary


def test_experiment_runner_with_scenario_c(mock_llm, tmp_path):
    """ExperimentRunner handles Scenario C with isolated memory."""
    dataset = [
        CVJDPair(
            pair_id="test",
            cv_text="Python ML developer",
            jd_text="ML engineer position",
        ),
    ]

    config = ExperimentConfig(
        scenarios=["A", "C"],
        output_dir=str(tmp_path),
        experiment_name="test_c",
    )

    runner = ExperimentRunner(mock_llm, config)
    report = runner.run_all(dataset)

    assert len(report.pair_results) == 1
    assert "score_A" in report.pair_results[0]
    assert "score_C" in report.pair_results[0]


# ── Test 4: Dataset loading ──────────────────────────────────

def test_load_dataset_from_json(tmp_path):
    """load_dataset_from_json parses the expected format."""
    dataset_file = tmp_path / "test_dataset.json"
    dataset_file.write_text(json.dumps({
        "pairs": [
            {
                "pair_id": "p1",
                "cv_text": "Some CV",
                "jd_text": "Some JD",
                "ground_truth_score": 70.0,
                "ground_truth_label": True,
                "description": "Test pair",
            },
            {
                "pair_id": "p2",
                "cv_text": "Another CV",
                "jd_text": "Another JD",
            },
        ]
    }), encoding="utf-8")

    pairs = load_dataset_from_json(str(dataset_file))

    assert len(pairs) == 2
    assert pairs[0].pair_id == "p1"
    assert pairs[0].ground_truth_score == 70.0
    assert pairs[0].ground_truth_label is True
    assert pairs[0].description == "Test pair"
    assert pairs[1].pair_id == "p2"
    assert pairs[1].ground_truth_score is None
    assert pairs[1].ground_truth_label is None


def test_load_actual_eval_dataset():
    """The actual eval_dataset.json file loads correctly."""
    pairs = load_dataset_from_json("data/eval_dataset.json")

    assert len(pairs) == 3
    assert pairs[0].pair_id == "strong_ml"
    assert pairs[1].pair_id == "partial_dev"
    assert pairs[2].pair_id == "weak_java"
    assert all(p.ground_truth_score is not None for p in pairs)
    assert pairs[0].ground_truth_label is True
    assert pairs[2].ground_truth_label is False


# ── Test 5: Report formatting ────────────────────────────────

def test_format_comparison_table():
    """format_comparison_table produces readable output."""
    report = ComparisonReport()

    report.add_pair(
        "pair_1",
        metrics_a=ScenarioMetrics(
            scenario="A", final_score=70, confidence=0.8,
            recommendation="good_match", reasoning_suggested_score=68,
        ),
        ground_truth_score=75.0,
    )
    report.compute_summary()

    table = format_comparison_table(report)

    assert "SCENARIO COMPARISON REPORT" in table
    assert "pair_1" in table
    assert "70.0" in table
    assert "SUMMARY STATISTICS" in table


def test_format_table_includes_classification():
    """format_comparison_table includes classification metrics when labels exist."""
    report = ComparisonReport()

    report.add_pair(
        "p1",
        metrics_a=ScenarioMetrics(scenario="A", final_score=70),
        ground_truth_score=80.0,
        ground_truth_label=True,
    )
    report.add_pair(
        "p2",
        metrics_a=ScenarioMetrics(scenario="A", final_score=30),
        ground_truth_score=20.0,
        ground_truth_label=False,
    )
    report.compute_summary()

    table = format_comparison_table(report)

    assert "CLASSIFICATION METRICS" in table
    assert "Prec." in table
    assert "Recall" in table
    assert "F1" in table


# ── Test 6: Results persistence ───────────────────────────────

def test_save_results(mock_llm, tmp_path):
    """ExperimentRunner saves results to JSON."""
    dataset = [
        CVJDPair(pair_id="p1", cv_text="cv", jd_text="jd", ground_truth_score=50),
    ]

    config = ExperimentConfig(
        scenarios=["A"],
        output_dir=str(tmp_path),
        experiment_name="save_test",
    )

    runner = ExperimentRunner(mock_llm, config)
    report = runner.run_all(dataset)
    path = runner.save_results(report)

    assert Path(path).exists()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["experiment"] == "save_test"
    assert len(data["pair_results"]) == 1
    assert "summary" in data


# ── Test 7: Classification metrics ────────────────────────────

def test_classification_perfect_predictions():
    """All predictions correct: precision=recall=F1=1.0."""
    predictions = [
        (80.0, True),   # match predicted, actually match   -> TP
        (70.0, True),   # match predicted, actually match   -> TP
        (30.0, False),  # no-match predicted, actually no   -> TN
        (20.0, False),  # no-match predicted, actually no   -> TN
    ]

    m = compute_classification_metrics(predictions, threshold=50.0)

    assert m.true_positives == 2
    assert m.true_negatives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.accuracy == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_classification_all_wrong():
    """All predictions wrong: precision=recall=0."""
    predictions = [
        (80.0, False),  # match predicted, actually no-match -> FP
        (30.0, True),   # no-match predicted, actually match -> FN
    ]

    m = compute_classification_metrics(predictions, threshold=50.0)

    assert m.true_positives == 0
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.accuracy == pytest.approx(0.0)
    assert m.precision == pytest.approx(0.0)
    assert m.recall == pytest.approx(0.0)
    assert m.f1 == pytest.approx(0.0)


def test_classification_mixed():
    """Mixed results: verifiable precision and recall."""
    predictions = [
        (80.0, True),   # TP
        (60.0, False),  # FP
        (40.0, True),   # FN
        (20.0, False),  # TN
    ]

    m = compute_classification_metrics(predictions, threshold=50.0)

    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.true_negatives == 1
    assert m.accuracy == pytest.approx(0.5)
    assert m.precision == pytest.approx(0.5)   # 1/(1+1)
    assert m.recall == pytest.approx(0.5)      # 1/(1+1)
    assert m.f1 == pytest.approx(0.5)          # 2*0.5*0.5/(0.5+0.5)


def test_classification_empty():
    """Empty input: all metrics 0."""
    m = compute_classification_metrics([], threshold=50.0)

    assert m.total == 0
    assert m.accuracy == 0.0
    assert m.f1 == 0.0


def test_classification_custom_threshold():
    """Threshold changes classification boundary."""
    predictions = [
        (65.0, True),   # match at threshold=50, no-match at threshold=70
    ]

    m_50 = compute_classification_metrics(predictions, threshold=50.0)
    m_70 = compute_classification_metrics(predictions, threshold=70.0)

    assert m_50.true_positives == 1   # 65 >= 50 → predicted match, is match → TP
    assert m_70.false_negatives == 1  # 65 < 70 → predicted no-match, is match → FN


def test_classification_in_comparison_report():
    """ComparisonReport.compute_summary also computes classification metrics."""
    report = ComparisonReport()

    report.add_pair(
        "p1",
        metrics_a=ScenarioMetrics(scenario="A", final_score=70),
        ground_truth_label=True,
    )
    report.add_pair(
        "p2",
        metrics_a=ScenarioMetrics(scenario="A", final_score=30),
        ground_truth_label=False,
    )

    report.compute_summary()

    assert "A" in report.classification
    cm = report.classification["A"]
    assert cm.true_positives == 1   # p1: score 70 >= 50, label True
    assert cm.true_negatives == 1   # p2: score 30 < 50, label False
    assert cm.accuracy == pytest.approx(1.0)


def test_ground_truth_label_derived_from_score():
    """When ground_truth_label is missing, it's derived from score >= 50."""
    report = ComparisonReport()

    # score=80 → derived label=True, score=30 → derived label=False
    report.add_pair(
        "p1",
        metrics_a=ScenarioMetrics(scenario="A", final_score=60),
        ground_truth_score=80.0,
        # no ground_truth_label → derived as True (80 >= 50)
    )
    report.add_pair(
        "p2",
        metrics_a=ScenarioMetrics(scenario="A", final_score=40),
        ground_truth_score=30.0,
        # no ground_truth_label → derived as False (30 < 50)
    )

    report.compute_summary()

    assert "A" in report.classification
    cm = report.classification["A"]
    assert cm.true_positives == 1   # predicted 60 >= 50, label True (from 80)
    assert cm.true_negatives == 1   # predicted 40 < 50, label False (from 30)


# ── Test 8: Trace recording and saving ────────────────────────

def test_runner_collects_traces(mock_llm, tmp_path):
    """ExperimentRunner collects TraceRecords for each run."""
    dataset = [
        CVJDPair(pair_id="p1", cv_text="cv text", jd_text="jd text"),
    ]

    config = ExperimentConfig(
        scenarios=["A", "B"],
        output_dir=str(tmp_path),
        experiment_name="trace_test",
    )

    runner = ExperimentRunner(mock_llm, config)
    runner.run_all(dataset)

    # 1 pair × 2 scenarios = 2 traces
    assert len(runner.traces) == 2

    # Check trace structure
    trace_a = runner.traces[0]
    assert trace_a.pair_id == "p1"
    assert trace_a.scenario == "A"
    assert trace_a.final_score > 0
    assert trace_a.recommendation != ""
    assert trace_a.explanation != ""
    assert len(trace_a.key_factors) > 0
    assert len(trace_a.strengths) > 0


def test_trace_to_dict(mock_llm, tmp_path):
    """TraceRecord.to_dict() produces expected structure."""
    dataset = [
        CVJDPair(pair_id="p1", cv_text="cv", jd_text="jd"),
    ]

    config = ExperimentConfig(
        scenarios=["A"],
        output_dir=str(tmp_path),
        experiment_name="dict_test",
    )

    runner = ExperimentRunner(mock_llm, config)
    runner.run_all(dataset)

    d = runner.traces[0].to_dict()

    assert "reasoning" in d
    assert "reflection" in d
    assert "decision" in d
    assert "strengths" in d["reasoning"]
    assert "gaps" in d["reasoning"]
    assert "is_consistent" in d["reflection"]
    assert "final_score" in d["decision"]
    assert "key_factors" in d["decision"]


def test_save_results_includes_traces(mock_llm, tmp_path):
    """save_results writes a separate traces JSON file."""
    dataset = [
        CVJDPair(
            pair_id="p1", cv_text="cv", jd_text="jd",
            ground_truth_score=70, ground_truth_label=True,
        ),
    ]

    config = ExperimentConfig(
        scenarios=["A"],
        output_dir=str(tmp_path),
        experiment_name="traces_save",
    )

    runner = ExperimentRunner(mock_llm, config)
    report = runner.run_all(dataset)
    metrics_path = runner.save_results(report)

    # Check metrics file has classification
    metrics_data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    assert "classification" in metrics_data
    assert "A" in metrics_data["classification"]
    assert "precision" in metrics_data["classification"]["A"]

    # Check traces file exists
    traces_path = tmp_path / "traces_save_traces.json"
    assert traces_path.exists()

    traces_data = json.loads(traces_path.read_text(encoding="utf-8"))
    assert len(traces_data) == 1
    assert traces_data[0]["pair_id"] == "p1"
    assert traces_data[0]["scenario"] == "A"
    assert "reasoning" in traces_data[0]
    assert "decision" in traces_data[0]
