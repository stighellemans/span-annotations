"""Example: convert multiple workflow annotations to Deduce annotations."""

from span_annotations.transform import workflow_annotations_to_deduce


def main() -> None:
    workflow_annotations = [
        {"begin": 0, "end": 4, "label": "Date", "text": "2001"},
        {"begin": 5, "end": 12, "label": "Contactdetails", "text": "a@b.com"},
    ]

    converted = workflow_annotations_to_deduce(workflow_annotations)
    for annotation in converted:
        print(annotation)


if __name__ == "__main__":
    main()
