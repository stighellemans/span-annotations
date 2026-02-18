# span-annotations

Utilities for span annotation parsing, alignment, extraction, and evaluation.

## What this package does

`span-annotations` helps you:

- Parse annotated HTML-like text (`<span class="...">...</span>`) into character offsets.
- Convert offset-based annotations back into tagged text.
- Align annotations from one text version to another when text differs slightly.
- Extract likely span matches from raw text using fuzzy alignment.
- Normalize/deduplicate spans for downstream annotation tools.
- Evaluate prediction quality at token level and generate edit operations to fix predictions.

## Installation

### 1. From GitHub

```bash
pip install git+https://github.com/stighellemans/span-annotations.git
```

### 2. Local editable install (recommended during development)

```bash
git clone https://github.com/stighellemans/span-annotations.git
cd span-annotations
pip install -e .
```

### 3. Verify install

```python
import span_annotations
from span_annotations import make_aligner

print(span_annotations.__version__)
print(make_aligner().mode)
```

## Requirements and runtime behavior

- Python `>=3.10`
- Dependencies are installed automatically:
  - `biopython`
  - `docdeid`
  - `jpype1`
  - `typing-extensions` (for older Python versions)

Important:

- `span_annotations.transform` and `span_annotations.evaluation` initialize a JVM on import using `jpype`.
- You need a working Java runtime (JRE/JDK) available in your environment for imports of those modules.
- JVM startup is guarded with `jpype.isJVMStarted()`, so repeated imports are safe.

## Annotation data model

Most functions use spans shaped like this:

```python
{
    "begin": 10,            # inclusive char offset
    "end": 24,              # exclusive char offset
    "label": "Name",       # label string
    "text": "Jan Janssens" # substring
}
```

Some functions also include:

- `Category`: label before `:` (for labels like `Name:Patient`)
- `Subtype`: label after `:`
- `similarity`: fuzzy match confidence from alignment

## Quickstart

```python
from span_annotations.transform import spans_to_annotations

original = "Patient Jan Janssens bezocht het ziekenhuis."
annotated = 'Patient <span class="Name">Jan Janssens</span> bezocht het ziekenhuis.'

spans = spans_to_annotations(annotated, original)
print(spans)
# [{"begin": 8, "end": 20, "label": "Name", "text": "Jan Janssens", ...}]
```

## Typical workflows

### A) Tagged text -> spans -> tagged text

```python
from span_annotations.transform import spans_to_annotations, annotations_to_spans

text = "Patient Jan Janssens bezocht het ziekenhuis."
annotated_text = 'Patient <span class="Name">Jan Janssens</span> bezocht het ziekenhuis.'

spans = spans_to_annotations(annotated_text, text)
roundtrip = annotations_to_spans(text, spans)

print(spans)
print(roundtrip)
```

Use this when:

- You receive model output with `<span class="...">` tags.
- You want storage/transport in offset-based JSON.

### B) Recover offsets when annotated and original text differ

```python
from span_annotations.transform import spans_to_annotations

original = "Patient Jan Janssens bezocht het ziekenhuis op 12/01/2024."
annotated = 'Patient <span class="Name">Jan Janssens</span> bezocht het ziekenhuis op 12/01/2024.'

# If cleaned annotated text differs from original, alignment remapping is applied automatically.
spans = spans_to_annotations(annotated, original)
```

Use this when:

- Formatting changes, extra whitespace, or mild text drift exists between two versions.

### C) Extract repeated fuzzy matches from phrase list

```python
from span_annotations.transform import extract_dict_spans

raw = "Jan Janssens werd gezien. Janssens werd opnieuw vermeld."
annotations = [
    {"annotated_text": "Janssens", "label": "Name:Patient"}
]

matches = extract_dict_spans(raw, annotations, min_similarity=0.85)
print(matches)
```

Use this when:

- You have phrase+label pairs and want all likely occurrences in a longer text.

### D) Token-level evaluation

```python
from span_annotations.evaluation import eval_token_level_io

text = "Jan Janssens werd geboren op 15/12/1984."
gold = [
    {"begin": 0, "end": 12, "label": "Name", "text": "Jan Janssens"},
    {"begin": 31, "end": 35, "label": "Age_Birthdate", "text": "1984"},
]
pred = [
    {"begin": 0, "end": 12, "label": "Name", "text": "Jan Janssens"},
    {"begin": 31, "end": 35, "label": "Name", "text": "1984"},
]

report = eval_token_level_io(text, gold, pred, lang="nl", verbose=True)
print(report["metrics"]["micro"])
print(report["metrics"]["per_type"])
```

Use this when:

- You need micro precision/recall/F1 and per-label metrics.
- You also want verbose confusion buckets (`TP`, `FP_plain`, `FP_mislabeled`, etc.).

### E) Evaluate annotation effort (add/delete/edit counts)

```python
from span_annotations.evaluation import evaluate_span_edits

gold = [{"begin": 0, "end": 12, "label": "Name", "text": "Jan Janssens"}]
pred = [{"begin": 0, "end": 12, "label": "ID", "text": "Jan Janssens"}]

ops = evaluate_span_edits(pred, gold)
print(ops["counts"])
print(ops["operations"])
```

Use this when:

- You want an operation-based quality measure for predicted annotations.
- Lower counts mean predictions are closer to gold annotations.

### F) Convert between Deduce and workflow annotations

```python
import docdeid as dd
from span_annotations.transform import (
    deduce_annotation_to_workflow,
    workflow_annotation_to_deduce,
)

deduce_ann = dd.Annotation(
    text="AZ Monica",
    start_char=783,
    end_char=792,
    tag="ziekenhuis",
)
workflow_ann = deduce_annotation_to_workflow(deduce_ann)
roundtrip = workflow_annotation_to_deduce(workflow_ann)

print(workflow_ann)
print(roundtrip)
```

Use this when:

- You need bidirectional conversion between `docdeid.Annotation` and workflow span dicts.
- You want a stable mapping between Deduce tags and workflow labels.

## API reference

### `span_annotations`

- `make_aligner(type="local")`
- `align_texts(text1, text2)`

### `span_annotations.alignment`

#### `make_aligner(type="local")`

Creates and returns a `Bio.Align.PairwiseAligner` configured with:

- `match_score = 2`
- `mismatch_score = -2`
- `open_gap_score = -0.5`
- `extend_gap_score = -0.1`

`type` is forwarded to aligner mode (`"local"` or `"global"`).

#### `align_texts(text1, text2)`

Runs pairwise alignment between two texts and returns the best alignment.

### `span_annotations.transform`

#### `labeled_to_annotations(text, labeled_text)`

Parses legacy placeholder format like:

- `"Patient [NAME] was seen on [DATE]"`

and maps placeholders back to character offsets in `text`.

#### `remove_html_spans(text)`

Removes complete and trailing broken `<span>` tags, keeping plain text content.

#### `spans_to_annotations(annotated_text, original_text)`

Main parser for `<span class="Label">text</span>` format.

Behavior:

- Parses tagged spans into offset annotations.
- If cleaned tagged text is different from `original_text`, it aligns and remaps offsets.
- Returns normalized span dictionaries (with `Category`/`Subtype` enrichment).

#### `annotations_to_spans(text, annotations)`

Converts offset annotations into inline `<span class="...">...</span>` text.

#### `is_premature_stop_aligned(annotated_text, original_text, strict_prefix=True)`

Checks whether cleaned annotated text appears to stop before the full original text.
Returns:

- `(premature: bool, covered_len: int)`

#### `normalize_to_inception_tokens(text, spans, lang="nl")`

Snaps span boundaries to nearest token boundaries (using Java `BreakIterator`) and deduplicates.
Useful for annotation tools expecting token-aligned offsets.

#### `transform_spans_inception(span_annotation)`

Adds:

- `Category` from `label.split(":")[0]`
- optional `Subtype` from `label.split(":")[1]`

#### `salvage_json_list(json_str)`

Best-effort recovery of complete elements from a truncated JSON list string.
Returns a valid JSON list string.

#### `parse_json_list(json_str)`

Parses a JSON list; if malformed/truncated, uses salvage logic and returns recovered elements.
Raises `ValueError` only if nothing list-like is recoverable.

#### `find_all_occurrences(raw_text, phrase, aligner=None, min_similarity=0.85, case_insensitive=True, max_hits=None)`

Iteratively finds non-overlapping approximate matches of `phrase` in `raw_text` using alignment.
Returns tuples:

- `(begin, end, similarity)`

#### `extract_dict_spans(raw_text, annotations, min_similarity=0.85, aligner=None)`

`annotations` format:

```python
[{"annotated_text": "...", "label": "Name:Patient"}]
```

Returns matched span dictionaries in text order.

#### `extract_tuple_spans(raw_text, annotations, min_similarity=0.85, aligner=None)`

Tuple/list variant of extraction input:

```python
[("phrase", "Label"), ...]
```

#### `trim_span(s, text)`

Trims leading/trailing whitespace from one span and updates `begin/end/text`.

#### `normalize_spans(spans, text)`

Applies trimming to all spans and drops empty spans.

#### `deduplicate_spans(spans)`

Deduplicates by exact `(begin, end)` and removes strictly contained spans.

#### Conversion functions exposed via `transform`

These functions are available from both:

- `span_annotations.transform` (same import style as the rest of the package)
- `span_annotations.conversion`

##### `workflow_label_to_deduce_tag(label, label_to_tag=..., strict=False)`

Maps a workflow label (e.g. `Name:Patient`) to a Deduce tag (e.g. `patient`).
If unknown and `strict=False`, returns the input label unchanged.

Example script:
- `examples/example_workflow_label_to_deduce_tag.py`

##### `deduce_annotation_to_workflow(annotation, source_text=None, deduce_to_label=..., strict=False)`

Converts one Deduce annotation (`docdeid.Annotation` or dict-like) into workflow format:

- `begin`, `end`, `label`, `text`, `Category`, `Subtype`

Example script:
- `examples/example_deduce_annotation_to_workflow.py`

##### `deduce_annotations_to_workflow(annotations, source_text=None, deduce_to_label=..., strict=False)`

Batch version of `deduce_annotation_to_workflow`.

Example script:
- `examples/example_deduce_annotations_to_workflow.py`

##### `workflow_annotation_to_deduce(annotation, source_text=None, label_to_tag=..., strict=False)`

Converts one workflow annotation dict into `docdeid.Annotation`.

Example script:
- `examples/example_workflow_annotation_to_deduce.py`

##### `workflow_annotations_to_deduce(annotations, source_text=None, label_to_tag=..., strict=False)`

Batch version of `workflow_annotation_to_deduce`.

Example script:
- `examples/example_workflow_annotations_to_deduce.py`

### `span_annotations.evaluation`

#### `segment_tokens(text, lang="nl")`

Tokenizes text using Java `BreakIterator` and returns token strings.

#### `tokens_with_offsets(text, lang="nl")`

Tokenizes and returns tuples:

- `(token, begin, end)`

#### `spans_to_io(tokens, spans, label_key="label", min_char_overlap=1, min_prop_overlap=0.0)`

Projects character spans to token labels in IO scheme:

- `"O"` for outside
- label string for entity tokens

#### `token_level_metrics_io(y_true, y_pred)`

Computes micro precision/recall/F1 and per-type metrics from IO labels.

#### `verbose_report(text, tokens, y_true, y_pred)`

Builds detailed token-level and span-level confusion buckets.

#### `summarize_per_type(per_type)`

Compacts verbose token confusion buckets into per-label metrics + breakdown.

#### `eval_token_level_io(...)`

End-to-end evaluation pipeline:

1. tokenization
2. span->token IO projection
3. metric computation
4. optional verbose diagnostics

#### `evaluate_span_edits(pred_spans, gold_spans, label_key="label")`

Computes an operation-distance style evaluation signal based on minimal exact-offset differences between prediction and gold:

- `Addition`
- `Deletion`
- `Edit` (same offset, different label)

The `counts` object is the main metric output; it quantifies annotation mismatch/effort (smaller is better).

## Input/output examples

### Span format with subtype labels

```python
{
    "begin": 42,
    "end": 53,
    "label": "Name:Patient",
    "text": "Jan Janssens",
    "Category": "Name",
    "Subtype": "Patient"
}
```

### `evaluate_span_edits` result shape

```python
{
    "counts": {
        "Addition": 1,
        "Deletion": 0,
        "Edit": 1,
        "total_ops": 2,
    },
    "operations": [
        {
            "op": "Edit",
            "begin": 10,
            "end": 20,
            "from": "ID",
            "to": "Name",
            "text": "Jan Janssens",
        },
        {
            "op": "Addition",
            "begin": 30,
            "end": 40,
            "label": "Date",
            "text": "12/01/2024",
        },
    ],
}
```

## Error handling notes

- `labeled_to_annotations` raises `ValueError` if template and source text do not match.
- `spans_to_annotations` may raise detailed `ValueError`s from remapping when alignment cannot produce contiguous valid mappings.
- `parse_json_list` raises `ValueError` if the string does not contain recoverable list data.
- Workflow bridge conversion functions raise `KeyError` in `strict=True` mode for unknown labels/tags.
- Workflow bridge conversion functions raise `ValueError` when required fields (`begin/end`, label/category, text/source text) are missing.

## Performance notes

- Fuzzy extraction (`find_all_occurrences`) is iterative and alignment-based; large texts + low thresholds increase runtime.
- If you repeatedly evaluate many documents, instantiate and reuse one aligner where possible.

## Troubleshooting

### `ImportError` / Java issues

Symptoms:

- import of `span_annotations.transform` or `span_annotations.evaluation` fails.

Checks:

1. Verify Java is installed and discoverable (`java -version`).
2. Ensure Python can import `jpype`.
3. Restart your Python process if JVM state got corrupted in an interactive session.

### Offsets look wrong after remapping

Checks:

1. Compare cleaned annotated text (`remove_html_spans`) with original text.
2. Validate that annotation `text` matches `text[begin:end]` before remapping.
3. Increase data quality first (normalization, whitespace cleanup) before lowering similarity thresholds.

## Minimal imports by task

- Alignment only:

```python
from span_annotations import make_aligner, align_texts
```

- HTML span parsing:

```python
from span_annotations.transform import spans_to_annotations, annotations_to_spans
```

- Evaluation:

```python
from span_annotations.evaluation import eval_token_level_io, evaluate_span_edits
```

- Workflow bridge conversion:

```python
from span_annotations.transform import (
    deduce_annotation_to_workflow,
    deduce_annotations_to_workflow,
    workflow_annotation_to_deduce,
    workflow_annotations_to_deduce,
    workflow_label_to_deduce_tag,
)
```

## Development

Run editable install:

```bash
pip install -e .
```

Current package layout:

- `span_annotations/alignment.py`
- `span_annotations/transform.py`
- `span_annotations/evaluation.py`
- `span_annotations/conversion.py`
- `tests/test_conversion.py`
- `examples/`
