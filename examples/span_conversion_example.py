"""Example script for standalone span/Deduce conversion."""

import docdeid as dd

from span_annotations.transform import (
    deduce_annotations_to_spans,
    spans_to_deduce_annotations,
)


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

    span_annotations = deduce_annotations_to_spans(deduce_annotations)
    print("Deduce -> span")
    for annotation in span_annotations:
        print(annotation)

    roundtrip = spans_to_deduce_annotations(span_annotations)
    print("\nSpan -> Deduce")
    for annotation in roundtrip:
        print(annotation)


if __name__ == "__main__":
    main()
