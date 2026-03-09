"""Example: convert multiple span annotations to Deduce annotations."""

from span_annotations.transform import spans_to_deduce_annotations


def main() -> None:
    span_annotations = [
        {"begin": 0, "end": 4, "label": "Date", "text": "2001"},
        {"begin": 5, "end": 12, "label": "Contactdetails", "text": "a@b.com"},
    ]

    converted = spans_to_deduce_annotations(span_annotations)
    for annotation in converted:
        print(annotation)


if __name__ == "__main__":
    main()
