import json

import pytest

from span_annotations.alignment import make_aligner

try:
    from span_annotations import transform
except Exception as exc:  # pragma: no cover - environment-dependent JVM availability
    pytest.skip(
        f"transform import unavailable in this environment: {exc}",
        allow_module_level=True,
    )


class _FakeAlignment:
    def __init__(self, indices):
        self.indices = indices


def test_labeled_to_annotations_success():
    text = "Patient Jan Jansen was seen on 2024-01-01."
    labeled = "Patient [NAME] was seen on [DATE]."
    out = transform.labeled_to_annotations(text, labeled)
    assert out == [
        {"label": "NAME", "begin": 8, "end": 18, "text": "Jan Jansen"},
        {"label": "DATE", "begin": 31, "end": 41, "text": "2024-01-01"},
    ]


def test_labeled_to_annotations_template_mismatch_raises():
    with pytest.raises(ValueError):
        transform.labeled_to_annotations("abc", "x [NAME]")


def test_is_premature_stop_aligned_equal_text():
    premature, covered_len = transform.is_premature_stop_aligned("abc", "abc")
    assert bool(premature) is False
    assert covered_len == 3


def test_is_premature_stop_aligned_truncated_prefix():
    premature, covered_len = transform.is_premature_stop_aligned("abc", "abcdef")
    assert bool(premature) is True
    assert covered_len == 3


def test_remove_html_spans_and_strip_trailing_broken():
    text = 'Hi <span class="Name">Jan</span>!'
    assert transform.remove_html_spans(text) == "Hi Jan!"
    assert transform.remove_html_spans("abc<span class='X'") == "abc"
    assert transform.strip_trailing_broken_spans("abc</span") == "abc"


def test_extract_class_label():
    assert (
        transform._extract_class_label('<span class="Name:Patient">') == "Name:Patient"
    )
    assert transform._extract_class_label("<span id='x'>") is None


def test_spans_to_raw_annotations_handles_nested_and_dangling():
    text = (
        'A <span class="Outer:Top">B <span class="Inner">C</span>D</span> E'
        "<span class='Tail'>X"
    )
    out = transform._spans_to_raw_annotations(text)
    labels = {item["label"] for item in out}
    assert labels == {"Inner", "Outer:Top", "Tail"}
    assert any(
        item["Category"] == "Outer" and item.get("Subtype") == "Top" for item in out
    )


def test_remap_annotations_to_original_success():
    alignment = _FakeAlignment(indices=[[0, 1, 2], [0, 1, 2]])
    annotations = [{"begin": 0, "end": 3, "label": "X", "text": "abc"}]
    out = transform.remap_annotations_to_original(alignment, annotations, "abc", "abc")
    assert out == [{"begin": 0, "end": 3, "label": "X", "text": "abc"}]


def test_remap_annotations_to_original_invalid_shape_raises():
    alignment = _FakeAlignment(indices=[[0, 1, 2]])
    with pytest.raises(ValueError):
        transform.remap_annotations_to_original(alignment, [], "abc", "abc")


def test_spans_to_annotations_without_remap():
    annotated = 'A <span class="Name">BC</span> D'
    original = "A BC D"
    out = transform.spans_to_annotations(annotated, original)
    assert out == [
        {
            "begin": 2,
            "end": 4,
            "label": "Name",
            "text": "BC",
            "Category": "Name",
        }
    ]


def test_spans_to_annotations_with_remap_when_text_differs():
    annotated = '<span class="Name">abc</span>'
    original = "xabc"
    out = transform.spans_to_annotations(annotated, original)
    assert out == [{"begin": 1, "end": 4, "label": "Name", "text": "abc"}]


def test_annotations_to_spans():
    text = "abcdef"
    annotations = [
        {"begin": 2, "end": 4, "label": "MID"},
        {"begin": 0, "end": 1, "label": "HEAD"},
    ]
    out = transform.annotations_to_spans(text, annotations)
    assert out == ('<span class="HEAD">a</span>b<span class="MID">cd</span>ef')


def test_normalize_to_inception_tokens_snaps_and_cleans_tie_breaker_key():
    text = "Jan Janssens."
    spans = [{"begin": 1, "end": 10, "label": "Name", "text": "an Janssen"}]
    out = transform.normalize_to_inception_tokens(text, spans, lang="en")
    assert len(out) == 1
    assert out[0]["begin"] == 0
    assert out[0]["end"] == 12
    assert "_orig_begin" not in out[0]


def test_transform_spans_inception_adds_category_and_subtype():
    spans = [
        {"begin": 0, "end": 1, "label": "Date", "text": "2"},
        {"begin": 2, "end": 5, "label": "Name:Patient", "text": "Jan"},
    ]
    out = transform.transform_spans_inception(spans)
    assert out[0]["Category"] == "Date"
    assert "Subtype" not in out[0]
    assert out[1]["Category"] == "Name"
    assert out[1]["Subtype"] == "Patient"


def test_salvage_json_list_and_parse_json_list():
    truncated = '[{"a": 1}, {"b": 2'
    fixed = transform.salvage_json_list(truncated)
    assert json.loads(fixed) == [{"a": 1}]

    parsed = transform.parse_json_list(truncated)
    assert parsed == [{"a": 1}]

    assert transform.salvage_json_list(123) is None


def test_parse_json_list_non_list_raises():
    with pytest.raises(ValueError):
        transform.parse_json_list('{"a": 1}')


def test_span_from_alignment_and_similarity():
    aligner = make_aligner()
    al = aligner.align("abc", "abc")[0]
    assert transform._span_from_alignment(al) == (0, 3)
    assert transform._similarity(al, query_len=3, match_score=2.0) == pytest.approx(1.0)


def test_choose_mask_char_and_fallback():
    assert transform._choose_mask_char("abc") in (
        "\x00",
        "\uffff",
        "\U0010ffff",
        "\x01",
    )
    phrase_with_all = "\x00\uffff\U0010ffff\x01"
    assert transform._choose_mask_char(phrase_with_all) == "\u25a0"


def test_find_all_occurrences_basic_case_insensitive_and_max_hits():
    raw = "abc xyz Abc"
    hits_default = transform.find_all_occurrences(raw, "abc", min_similarity=1.0)
    assert [(b, e) for b, e, _ in hits_default] == [(0, 3), (8, 11)]

    hits_case_sensitive = transform.find_all_occurrences(
        raw, "ABC", min_similarity=1.0, case_insensitive=False
    )
    assert hits_case_sensitive == []

    hits_limited = transform.find_all_occurrences(
        raw, "abc", min_similarity=1.0, max_hits=1
    )
    assert len(hits_limited) == 1


def test_extract_dict_spans_and_extract_tuple_spans():
    raw = "Jan Jan"
    dict_out = transform.extract_dict_spans(
        raw_text=raw,
        annotations=[{"annotated_text": "Jan", "label": "Name:Patient"}],
        min_similarity=1.0,
    )
    assert len(dict_out) == 2
    assert dict_out[0]["Category"] == "Name"
    assert dict_out[0]["Subtype"] == "Patient"

    tuple_out = transform.extract_tuple_spans(
        raw_text=raw,
        annotations=[("Jan", "Name")],
        min_similarity=1.0,
    )
    assert len(tuple_out) == 2
    assert tuple_out[0]["Category"] == "Name"


def test_trim_normalize_and_deduplicate_spans():
    text = "  Jan  "
    span = {"begin": 0, "end": 7, "label": "Name", "text": text}
    trimmed = transform.trim_span(dict(span), text)
    assert trimmed == {"begin": 2, "end": 5, "label": "Name", "text": "Jan"}

    normalized = transform.normalize_spans(
        [{"begin": 0, "end": 2, "label": "W", "text": "  "}, span], text
    )
    assert normalized == [{"begin": 2, "end": 5, "label": "Name", "text": "Jan"}]

    deduped = transform.deduplicate_spans(
        [
            {"begin": 0, "end": 10, "label": "Outer", "_orig_begin": 0},
            {"begin": 0, "end": 10, "label": "Outer2", "_orig_begin": 1},
            {"begin": 2, "end": 5, "label": "Inner", "_orig_begin": 2},
        ]
    )
    assert deduped == [{"begin": 0, "end": 10, "label": "Outer"}]


def test_transform_re_exports_conversion_api():
    from span_annotations import deduce

    assert transform.workflow_label_to_deduce_tag is deduce.workflow_label_to_deduce_tag
    assert (
        transform.workflow_annotation_to_deduce is deduce.workflow_annotation_to_deduce
    )
    assert (
        transform.deduce_annotation_to_workflow is deduce.deduce_annotation_to_workflow
    )
