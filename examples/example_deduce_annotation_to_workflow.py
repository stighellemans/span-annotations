"""Example: convert one Deduce annotation to workflow format."""

import docdeid as dd

from span_annotations.transform import deduce_annotation_to_workflow


def main() -> None:
    annotation = dd.Annotation(
        text="AZ Monica",
        start_char=783,
        end_char=792,
        tag="ziekenhuis",
    )

    converted = deduce_annotation_to_workflow(annotation)
    print(converted)


if __name__ == "__main__":
    main()
