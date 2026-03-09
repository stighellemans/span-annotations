"""Example: map a span label to a Deduce tag."""

from span_annotations.transform import span_label_to_deduce_tag


def main() -> None:
    print(span_label_to_deduce_tag("Name:Patient"))
    print(span_label_to_deduce_tag("Date"))
    print(span_label_to_deduce_tag("CustomLabel", strict=False))


if __name__ == "__main__":
    main()
