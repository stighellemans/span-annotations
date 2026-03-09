"""Example: convert one span annotation to a Deduce annotation."""

from span_annotations.transform import span_to_deduce_annotation


def main() -> None:
    annotation = {
        "begin": 783,
        "end": 792,
        "label": "Organization:Healthcare",
        "text": "AZ Monica",
    }

    converted = span_to_deduce_annotation(annotation)
    print(converted)


if __name__ == "__main__":
    main()
