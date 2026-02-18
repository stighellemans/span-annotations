"""Example: convert multiple Deduce annotations to workflow format."""

import docdeid as dd

from span_annotations.transform import deduce_annotations_to_workflow


def main() -> None:
    deduce_annotations = [
        dd.Annotation(
            text="AZ Monica",
            start_char=783,
            end_char=792,
            tag="ziekenhuis",
        ),
        dd.Annotation(
            text="J. Jansen",
            start_char=64,
            end_char=73,
            tag="persoon",
        ),
    ]

    converted = deduce_annotations_to_workflow(deduce_annotations)
    for annotation in converted:
        print(annotation)


if __name__ == "__main__":
    main()
