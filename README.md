# span-annotations

Utilities for span annotation parsing, alignment, and evaluation.

## Install

```bash
pip install git+https://github.com/stighellemans/span-annotations.git
```

For local development:

```bash
pip install -e .
```

## Usage

```python
from span_annotations.span_labels import spans_to_annotations
from span_annotations.evaluation import eval_token_level_io
from span_annotations import make_aligner

annotated = 'Patient <span class="Name">Jan Janssens</span> bezocht het ziekenhuis.'
original = 'Patient Jan Janssens bezocht het ziekenhuis.'

spans = spans_to_annotations(annotated, original)
print(spans)
```

## Notes

- Some tokenization helpers use Java `BreakIterator` via `jpype`, so a local Java runtime is required for those functions.
