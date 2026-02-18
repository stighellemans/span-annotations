"""Example script for standalone workflow/Deduce conversion."""

import docdeid as dd

from span_annotations.transform import (
    deduce_annotations_to_workflow,
    workflow_annotations_to_deduce,
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

    workflow_annotations = deduce_annotations_to_workflow(deduce_annotations)
    print("Deduce -> workflow")
    for annotation in workflow_annotations:
        print(annotation)

    roundtrip = workflow_annotations_to_deduce(workflow_annotations)
    print("\nWorkflow -> Deduce")
    for annotation in roundtrip:
        print(annotation)


if __name__ == "__main__":
    main()
