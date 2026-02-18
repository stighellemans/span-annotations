"""Example: convert one workflow annotation to a Deduce annotation."""

from span_annotations.transform import workflow_annotation_to_deduce


def main() -> None:
    annotation = {
        "begin": 783,
        "end": 792,
        "label": "Organization:Healthcare",
        "text": "AZ Monica",
    }

    converted = workflow_annotation_to_deduce(annotation)
    print(converted)


if __name__ == "__main__":
    main()
