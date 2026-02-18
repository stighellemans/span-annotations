"""Example: map a workflow label to a Deduce tag."""

from span_annotations.transform import workflow_label_to_deduce_tag


def main() -> None:
    print(workflow_label_to_deduce_tag("Name:Patient"))
    print(workflow_label_to_deduce_tag("Date"))
    print(workflow_label_to_deduce_tag("CustomLabel", strict=False))


if __name__ == "__main__":
    main()
