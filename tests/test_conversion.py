import docdeid as dd
import pytest

from span_annotations import deduce
from span_annotations.deduce import (
    DEFAULT_DEDUCE_TO_WORKFLOW_LABEL,
    DEFAULT_WORKFLOW_TO_DEDUCE_TAG,
    deduce_annotation_to_workflow,
    deduce_annotations_to_workflow,
    workflow_annotation_to_deduce,
    workflow_annotations_to_deduce,
    workflow_label_to_deduce_tag,
)


class TestConstants:
    def test_default_mapping_constants_have_expected_examples(self):
        assert DEFAULT_DEDUCE_TO_WORKFLOW_LABEL["patient"] == "Name:Patient"
        assert (
            DEFAULT_DEDUCE_TO_WORKFLOW_LABEL["ziekenhuis"] == "Organization:Healthcare"
        )
        assert DEFAULT_WORKFLOW_TO_DEDUCE_TAG["Name:Patient"] == "patient"
        assert DEFAULT_WORKFLOW_TO_DEDUCE_TAG["Contactdetails"] == "telefoonnummer"

    def test_conversion_module_exports_match_package_exports(self):
        assert deduce.deduce_annotation_to_workflow is deduce_annotation_to_workflow
        assert deduce.workflow_annotation_to_deduce is workflow_annotation_to_deduce
        assert deduce.workflow_label_to_deduce_tag is workflow_label_to_deduce_tag


class TestWorkflowLabelToDeduceTag:
    def test_known_label(self):
        assert workflow_label_to_deduce_tag("Contactdetails") == "telefoonnummer"

    def test_unknown_label_non_strict_returns_label(self):
        assert (
            workflow_label_to_deduce_tag("UnknownLabel", strict=False) == "UnknownLabel"
        )

    def test_unknown_label_strict_raises(self):
        with pytest.raises(KeyError):
            workflow_label_to_deduce_tag("UnknownLabel", strict=True)

    def test_custom_mapping(self):
        mapping = {"X": "custom_tag"}
        assert workflow_label_to_deduce_tag("X", label_to_tag=mapping) == "custom_tag"


class TestDeduceAnnotationToWorkflow:
    def test_from_deduce_annotation(self):
        annotation = dd.Annotation(
            text="AZ Monica",
            start_char=783,
            end_char=792,
            tag="ziekenhuis",
            priority=7,
        )

        converted = deduce_annotation_to_workflow(annotation)

        assert converted == {
            "begin": 783,
            "end": 792,
            "label": "Organization:Healthcare",
            "text": "AZ Monica",
            "Category": "Organization",
            "Subtype": "Healthcare",
        }

    def test_from_mapping_uses_source_text_when_text_is_missing(self):
        source = "xxJan Jansenyy"
        annotation = {"begin": 2, "end": 12, "tag": "persoon"}

        converted = deduce_annotation_to_workflow(annotation, source_text=source)

        assert converted["text"] == "Jan Jansen"
        assert converted["label"] == "Name:Other"
        assert converted["Category"] == "Name"
        assert converted["Subtype"] == "Other"

    def test_unknown_tag_non_strict_falls_back_to_tag(self):
        annotation = dd.Annotation(text="x", start_char=0, end_char=1, tag="unknown")
        converted = deduce_annotation_to_workflow(annotation, strict=False)

        assert converted["label"] == "unknown"
        assert converted["Category"] == "unknown"
        assert converted["Subtype"] is None

    def test_unknown_tag_strict_raises(self):
        annotation = {"begin": 0, "end": 1, "tag": "unknown", "text": "x"}
        with pytest.raises(KeyError):
            deduce_annotation_to_workflow(annotation, strict=True)

    def test_missing_text_and_source_text_raises(self):
        annotation = {"begin": 0, "end": 1, "tag": "patient"}
        with pytest.raises(ValueError):
            deduce_annotation_to_workflow(annotation)


class TestDeduceAnnotationsToWorkflow:
    def test_multiple_annotations_conversion(self):
        annotations = [
            dd.Annotation(text="A", start_char=0, end_char=1, tag="patient"),
            {"start_char": 2, "end_char": 3, "tag": "datum", "text": "B"},
        ]

        converted = deduce_annotations_to_workflow(annotations)

        assert [annotation["label"] for annotation in converted] == [
            "Name:Patient",
            "Date",
        ]


class TestWorkflowAnnotationToDeduce:
    def test_from_label(self):
        annotation = {
            "begin": 783,
            "end": 792,
            "label": "Organization:Healthcare",
            "text": "AZ Monica",
            "priority": 4,
        }
        converted = workflow_annotation_to_deduce(annotation)

        assert converted == dd.Annotation(
            text="AZ Monica",
            start_char=783,
            end_char=792,
            tag="zorginstelling",
            priority=4,
        )

    def test_from_category_and_subtype(self):
        annotation = {
            "begin": 10,
            "end": 21,
            "Category": "Address_Location",
            "Subtype": "Patient",
            "text": "Main Street",
        }

        converted = workflow_annotation_to_deduce(annotation)

        assert converted == dd.Annotation(
            text="Main Street",
            start_char=10,
            end_char=21,
            tag="locatie",
            priority=0,
        )

    def test_from_category_without_subtype(self):
        annotation = {
            "begin": 0,
            "end": 4,
            "Category": "Date",
            "text": "2001",
        }

        converted = workflow_annotation_to_deduce(annotation)

        assert converted == dd.Annotation(
            text="2001",
            start_char=0,
            end_char=4,
            tag="datum",
            priority=0,
        )

    def test_uses_source_text_when_text_is_missing(self):
        text = "My id is 123456789."
        annotation = {"begin": 9, "end": 18, "label": "ID:Patient"}

        converted = workflow_annotation_to_deduce(annotation, source_text=text)

        assert converted == dd.Annotation(
            text="123456789",
            start_char=9,
            end_char=18,
            tag="id",
            priority=0,
        )

    def test_unknown_label_non_strict_falls_back_to_label(self):
        annotation = {"begin": 0, "end": 1, "label": "CustomLabel", "text": "x"}

        converted = workflow_annotation_to_deduce(annotation, strict=False)

        assert converted == dd.Annotation(
            text="x",
            start_char=0,
            end_char=1,
            tag="CustomLabel",
            priority=0,
        )

    def test_unknown_label_strict_raises(self):
        annotation = {"begin": 0, "end": 1, "label": "CustomLabel", "text": "x"}

        with pytest.raises(KeyError):
            workflow_annotation_to_deduce(annotation, strict=True)

    def test_missing_label_and_category_raises(self):
        annotation = {"begin": 0, "end": 1, "text": "x"}
        with pytest.raises(ValueError):
            workflow_annotation_to_deduce(annotation)

    def test_missing_text_and_source_text_raises(self):
        annotation = {"begin": 0, "end": 1, "label": "Date"}
        with pytest.raises(ValueError):
            workflow_annotation_to_deduce(annotation)


class TestWorkflowAnnotationsToDeduce:
    def test_multiple_workflow_annotations(self):
        annotations = [
            {"begin": 0, "end": 4, "label": "Date", "text": "2001"},
            {"begin": 5, "end": 12, "label": "Contactdetails", "text": "a@b.com"},
        ]

        converted = workflow_annotations_to_deduce(annotations)

        assert converted == [
            dd.Annotation(text="2001", start_char=0, end_char=4, tag="datum"),
            dd.Annotation(
                text="a@b.com",
                start_char=5,
                end_char=12,
                tag="telefoonnummer",
            ),
        ]
