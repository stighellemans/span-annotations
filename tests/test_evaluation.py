import pytest

try:
    from span_annotations import evaluation
except Exception as exc:  # pragma: no cover - environment-dependent JVM availability
    pytest.skip(
        f"evaluation import unavailable in this environment: {exc}",
        allow_module_level=True,
    )


def test_segment_tokens_and_tokens_with_offsets():
    text = "Jan zag Marie."
    tokens = evaluation.segment_tokens(text, lang="nl")
    assert "Jan" in tokens
    assert "Marie" in tokens

    with_offsets = evaluation.tokens_with_offsets(text, lang="nl")
    assert all(t == text[b:e] for t, b, e in with_offsets)
    assert all(b < e for _, b, e in with_offsets)


def test_spans_to_io_with_overlap_thresholds():
    tokens = [("Jan", 0, 3), ("Jansen", 4, 10), ("test", 11, 15)]
    spans = [{"begin": 0, "end": 10, "label": "Name"}]

    labels = evaluation.spans_to_io(tokens, spans)
    assert labels == ["Name", "Name", "O"]

    strict_labels = evaluation.spans_to_io(tokens, spans, min_prop_overlap=1.0)
    assert strict_labels == ["Name", "Name", "O"]


def test_token_level_metrics_io():
    y_true = ["Name", "O", "Date"]
    y_pred = ["Name", "Name", "O"]
    metrics = evaluation.token_level_metrics_io(y_true, y_pred)

    assert metrics["micro"]["precision"] == pytest.approx(0.5)
    assert metrics["micro"]["recall"] == pytest.approx(0.5)
    assert metrics["micro"]["f1"] == pytest.approx(0.5)
    assert metrics["micro"]["support"] == 2
    assert set(metrics["per_type"]) == {"Date", "Name"}


def test_runs_groups_consecutive_labels():
    tokens = [("a", 0, 1), ("b", 1, 2), ("c", 2, 3), ("d", 3, 4)]
    labels = ["X", "X", "O", "X"]
    runs = evaluation._runs(tokens, labels)
    assert runs == [
        {
            "label": "X",
            "i_start": 0,
            "i_end": 1,
            "char_begin": 0,
            "char_end": 2,
            "text": "ab",
        },
        {
            "label": "X",
            "i_start": 3,
            "i_end": 3,
            "char_begin": 3,
            "char_end": 4,
            "text": "d",
        },
    ]


def test_verbose_report_produces_expected_buckets():
    text = "Jan 2024"
    tokens = [("Jan", 0, 3), ("2024", 4, 8)]
    y_true = ["Name", "Date"]
    y_pred = ["Date", "Date"]

    report = evaluation.verbose_report(text, tokens, y_true, y_pred)
    per_type = report["token_level_per_type"]
    span_level = report["span_level"]

    assert per_type["Date"]["TP"] == [(1, "2024")]
    assert per_type["Date"]["FP_mislabeled"] == [(0, "Jan")]
    assert per_type["Name"]["FN_mislabeled"] == [(0, "Jan")]
    assert len(span_level["TP"]) == 1
    assert len(span_level["FN_mislabeled"]) == 1
    assert len(span_level["FP_plain"]) == 0


def test_summarize_per_type():
    per_type = {
        "Name": {
            "TP": [(0, "Jan")],
            "FP_plain": [(2, "x")],
            "FP_mislabeled": [],
            "FN_missed": [],
            "FN_mislabeled": [(1, "J")],
        }
    }
    summary = evaluation.summarize_per_type(per_type)
    assert summary["Name"]["support"] == 2
    assert summary["Name"]["breakdown"]["fp_plain"] == 1
    assert summary["Name"]["breakdown"]["fn_mislabeled"] == 1


def test_eval_token_level_io_verbose():
    text = "Jan 2024"
    gold = [{"begin": 0, "end": 3, "label": "Name", "text": "Jan"}]
    pred = [{"begin": 0, "end": 3, "label": "Name", "text": "Jan"}]

    out = evaluation.eval_token_level_io(text, gold, pred, verbose=True)
    assert out["tokens"][0] == "Jan"
    assert out["metrics"]["micro"]["f1"] == pytest.approx(1.0)
    assert "verbose" in out
    assert "token_level_per_type_summary" in out["verbose"]


def test_key_and_label_helpers():
    span = {"begin": "2", "end": "5", "label": "Name", "Category": "Name"}
    assert evaluation._key(span) == (2, 5)
    assert evaluation._label(span) == "Name"
    assert evaluation._label({"Category": "Date"}, label_key="Category") == "Date"


def test_evaluate_span_edits_counts_and_operations():
    pred = [
        {"begin": 0, "end": 3, "label": "ID", "text": "Jan"},
        {"begin": 4, "end": 8, "label": "Date", "text": "2024"},
    ]
    gold = [
        {"begin": 0, "end": 3, "label": "Name", "text": "Jan"},
        {"begin": 9, "end": 10, "label": "X", "text": "x"},
    ]

    result = evaluation.evaluate_span_edits(pred, gold)
    assert result["counts"] == {"Addition": 1, "Deletion": 1, "Edit": 1, "total_ops": 3}
    assert [op["op"] for op in result["operations"]] == ["Edit", "Addition", "Deletion"]
