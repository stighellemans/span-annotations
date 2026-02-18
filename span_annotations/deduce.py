"""Helpers to convert annotations between Deduce and workflow schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import docdeid as dd

__all__ = [
    "DEFAULT_DEDUCE_TO_WORKFLOW_LABEL",
    "DEFAULT_WORKFLOW_TO_DEDUCE_TAG",
    "workflow_label_to_deduce_tag",
    "deduce_annotation_to_workflow",
    "deduce_annotations_to_workflow",
    "workflow_annotation_to_deduce",
    "workflow_annotations_to_deduce",
]

# Default mapping from Deduce tags to workflow labels.
DEFAULT_DEDUCE_TO_WORKFLOW_LABEL = {
    "patient": "Name:Patient",
    "persoon": "Name:Other",
    "locatie": "Address_Location:Other",
    "datum": "Date",
    "leeftijd": "Age_Birthdate",
    "telefoonnummer": "Contactdetails",
    "emailadres": "Contactdetails",
    "url": "Contactdetails",
    "id": "ID:Patient",
    "bsn": "ID:Patient",
    "zorginstelling": "Organization:Healthcare",
    "ziekenhuis": "Organization:Healthcare",
}

# Default canonical mapping from workflow labels to Deduce tags.
DEFAULT_WORKFLOW_TO_DEDUCE_TAG = {
    "Address_Location:Caregiver": "locatie",
    "Address_Location:Patient": "locatie",
    "Address_Location:Other": "locatie",
    "Age_Birthdate": "leeftijd",
    "Anonymize_Other": "anonymize_other",
    "Contactdetails": "telefoonnummer",
    "Date": "datum",
    "ID:Caregiver": "id",
    "ID:Patient": "id",
    "Name:Caregiver": "persoon",
    "Name:Patient": "patient",
    "Name:Other": "persoon",
    "Organization:Healthcare": "zorginstelling",
    "Organization:Other": "zorginstelling",
    "Profession": "beroep",
}


def workflow_label_to_deduce_tag(
    label: str,
    label_to_tag: Mapping[str, str] = DEFAULT_WORKFLOW_TO_DEDUCE_TAG,
    strict: bool = False,
) -> str:
    """Map a workflow label to a Deduce tag."""

    tag = label_to_tag.get(label)
    if tag is not None:
        return tag
    if strict:
        raise KeyError(f"Workflow label is not in label_to_tag mapping: {label}")

    return label


def deduce_annotation_to_workflow(
    annotation: dd.Annotation | Mapping[str, Any],
    source_text: Optional[str] = None,
    deduce_to_label: Mapping[str, str] = DEFAULT_DEDUCE_TO_WORKFLOW_LABEL,
    strict: bool = False,
) -> dict[str, Any]:
    """Convert a Deduce annotation object/dict to workflow annotation format."""

    begin, end, tag, ann_text, _priority = _read_deduce_annotation(
        annotation, source_text=source_text
    )

    label = deduce_to_label.get(tag)
    if label is None:
        if strict:
            raise KeyError(f"Deduce tag is not in deduce_to_label mapping: {tag}")
        label = tag

    category, subtype = _split_label(label)

    return {
        "begin": begin,
        "end": end,
        "label": label,
        "text": ann_text,
        "Category": category,
        "Subtype": subtype,
    }


def deduce_annotations_to_workflow(
    annotations: Iterable[dd.Annotation | Mapping[str, Any]],
    source_text: Optional[str] = None,
    deduce_to_label: Mapping[str, str] = DEFAULT_DEDUCE_TO_WORKFLOW_LABEL,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Convert multiple Deduce annotations to workflow annotation dictionaries."""

    return [
        deduce_annotation_to_workflow(
            annotation=annotation,
            source_text=source_text,
            deduce_to_label=deduce_to_label,
            strict=strict,
        )
        for annotation in annotations
    ]


def workflow_annotation_to_deduce(
    annotation: Mapping[str, Any],
    source_text: Optional[str] = None,
    label_to_tag: Mapping[str, str] = DEFAULT_WORKFLOW_TO_DEDUCE_TAG,
    strict: bool = False,
) -> dd.Annotation:
    """Convert a workflow annotation dictionary to a Deduce ``Annotation``."""

    begin = _get_required_int(annotation, "begin", "start_char")
    end = _get_required_int(annotation, "end", "end_char")
    label = _build_label(annotation)
    ann_text = _get_annotation_text(
        annotation, begin=begin, end=end, source_text=source_text
    )
    priority = int(annotation.get("priority", 0))

    tag = workflow_label_to_deduce_tag(
        label=label,
        label_to_tag=label_to_tag,
        strict=strict,
    )

    return dd.Annotation(
        text=ann_text,
        start_char=begin,
        end_char=end,
        tag=tag,
        priority=priority,
    )


def workflow_annotations_to_deduce(
    annotations: Iterable[Mapping[str, Any]],
    source_text: Optional[str] = None,
    label_to_tag: Mapping[str, str] = DEFAULT_WORKFLOW_TO_DEDUCE_TAG,
    strict: bool = False,
) -> list[dd.Annotation]:
    """Convert multiple workflow annotation dictionaries to Deduce annotations."""

    return [
        workflow_annotation_to_deduce(
            annotation=annotation,
            source_text=source_text,
            label_to_tag=label_to_tag,
            strict=strict,
        )
        for annotation in annotations
    ]


def _read_deduce_annotation(
    annotation: dd.Annotation | Mapping[str, Any], source_text: Optional[str]
) -> tuple[int, int, str, str, int]:
    if isinstance(annotation, Mapping):
        begin = _get_required_int(annotation, "start_char", "begin")
        end = _get_required_int(annotation, "end_char", "end")
        tag = _get_required_str(annotation, "tag")
        text = _get_annotation_text(
            annotation, begin=begin, end=end, source_text=source_text
        )
        priority = int(annotation.get("priority", 0))
        return begin, end, tag, text, priority

    begin = int(annotation.start_char)
    end = int(annotation.end_char)
    tag = str(annotation.tag)
    text = str(annotation.text)
    priority = int(annotation.priority)

    return begin, end, tag, text, priority


def _split_label(label: str) -> tuple[str, Optional[str]]:
    if ":" in label:
        category, subtype = label.split(":", maxsplit=1)
        return category, subtype
    return label, None


def _build_label(annotation: Mapping[str, Any]) -> str:
    label = annotation.get("label")
    if label is not None:
        return str(label)

    category = annotation.get("Category")
    if category is None:
        raise ValueError("Annotation is missing `label` and `Category`.")

    subtype = annotation.get("Subtype")
    if subtype in (None, ""):
        return str(category)

    return f"{category}:{subtype}"


def _get_annotation_text(
    annotation: Mapping[str, Any],
    begin: int,
    end: int,
    source_text: Optional[str],
) -> str:
    ann_text = annotation.get("text")
    if ann_text is not None:
        return str(ann_text)

    if source_text is None:
        raise ValueError(
            "Annotation text is missing. Provide `text` in annotation or `source_text`."
        )

    return source_text[begin:end]


def _get_required_str(values: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return str(value)

    raise ValueError(f"Missing required field. Expected one of: {keys}")


def _get_required_int(values: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return int(value)

    raise ValueError(f"Missing required field. Expected one of: {keys}")
