from __future__ import annotations

import json
import re
from json import JSONDecoder
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import jpype
import jpype.imports
from Bio import Align
from jpype.types import *
from typing_extensions import NotRequired

from .alignment import align_texts, make_aligner

# Start JVM (point to your JDK if not on PATH)
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[])

from java.text import BreakIterator
from java.util import Locale

# --- Typed structures ---


class SpanRecord(TypedDict):
    begin: int
    end: int
    label: str
    text: str
    Category: str
    Subtype: NotRequired[str]


__all__ = [
    "SpanRecord",
    "labeled_to_annotations",
    "is_premature_stop_aligned",
    "remove_html_spans",
    "strip_trailing_broken_spans",
    "remap_annotations_to_original",
    "spans_to_annotations",
    "annotations_to_spans",
    "normalize_to_inception_tokens",
    "split_label_into_category_subtype",
    "salvage_json_list",
    "parse_json_list",
    "find_all_occurrences",
    "extract_dict_spans",
    "extract_tuple_spans",
    "trim_span",
    "normalize_spans",
    "deduplicate_spans",
    "DEFAULT_DEDUCE_TO_WORKFLOW_LABEL",
    "DEFAULT_WORKFLOW_TO_DEDUCE_TAG",
    "workflow_label_to_deduce_tag",
    "deduce_annotation_to_workflow",
    "deduce_annotations_to_workflow",
    "workflow_annotation_to_deduce",
    "workflow_annotations_to_deduce",
]


# ------- Old labeling format: text replaced by [LABEL_NAME] ------- #
def labeled_to_annotations(text: str, labeled_text: str):
    tokens = re.split(r"(\[[A-Z_]+?\])", labeled_text)
    pattern_parts, labels = [], []

    for tok in tokens:
        if not tok:
            continue
        m = re.fullmatch(r"\[([A-Z_]+?)\]", tok)
        if m:
            labels.append(m.group(1))
            pattern_parts.append("(.+?)")  # non-greedy capture
        else:
            placeholder = "<<<WS>>>"
            s = re.sub(r"\s+", placeholder, tok)
            s = re.escape(s).replace(re.escape(placeholder), r"\s+")
            pattern_parts.append(s)

    pattern = "".join(pattern_parts)
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError("Template did not match the text.")

    spans = []
    for i, base in enumerate(labels, 1):
        start, end = match.span(i)
        spans.append(
            {
                "label": base,
                "begin": start,
                "end": end,
                "text": text[start:end],
            }
        )
    return spans


def is_premature_stop_aligned(
    annotated_text: str,
    original_text: str,
    strict_prefix: bool = True,
) -> Tuple[bool, int]:
    """
    Local-align version.
    Returns (premature, covered_len, alignment).

    Rules:
      - Strip <span ...> from annotated_text.
      - Accept only alignments that start at 0 in BOTH strings and
        consume the entire cleaned query (i.e., q_end == len(cleaned)).
      - If strict_prefix=True, also require exact prefix equality on the covered region.
    """
    cleaned = remove_html_spans(annotated_text)

    # Trivial fast paths
    if cleaned == original_text:
        return (False, len(original_text))
    if cleaned == "":
        return (False, 0)

    aligner = make_aligner("local")

    # Try to find a local alignment that anchors at start and consumes all of `cleaned`
    best_ok: Optional[Align.Alignment] = None
    for aln in aligner.align(cleaned, original_text):
        q_blocks, t_blocks = aln.aligned
        if q_blocks.size == 0 or t_blocks.size == 0:
            continue
        # anchor at start of both query and target
        if q_blocks[0, 0] != 0 or t_blocks[0, 0] != 0:
            continue
        # consume entire cleaned query
        if q_blocks[-1, 1] != len(cleaned):
            continue
        # Optional: enforce exact prefix equality
        covered_len = t_blocks[-1, 1]
        if strict_prefix and original_text[: len(cleaned)] != cleaned:
            continue

        best_ok = aln
        break  # first best is fine (Aligner already sorts by score)

    if best_ok is None:
        # No acceptable start-anchored full-query local alignment found.
        # Fall back to reporting zero coverage (not premature).
        return (False, 0)

    # Compute covered length from the accepted alignment
    q_blocks, t_blocks = best_ok.aligned
    covered_len = t_blocks[-1, 1]
    premature = covered_len < len(original_text)
    return (premature, covered_len)


# ------ New labeling format: text enclosed in <span class="...">...</span>tags ------ #
def remove_html_spans(text: str):
    # 1) Remove complete <span ...> and </span> tags (case-insensitive)
    text = re.sub(r"</?\s*span\b[^>]*>", "", text, flags=re.I)
    # 2) Remove truncated <span ...   or   </span ...  at the END of the string
    text = re.sub(r"</?\s*span\b[^>]*\Z", "", text, flags=re.I)
    # 3) Remove any dangling tag-start at the END (e.g., '<', '</')
    text = re.sub(r"</?\s*\Z", "", text)
    return text


OPEN_TAG_RE = re.compile(r"<span\b[^>]*>", re.IGNORECASE)
CLOSE_TAG_RE = re.compile(r"</span\s*>", re.IGNORECASE)
CLASS_ATTR_RE = re.compile(r'class\s*=\s*(?:"([^"]*)"|\'([^\']*)\')', re.IGNORECASE)


def _extract_class_label(tag: str) -> Optional[str]:
    m = CLASS_ATTR_RE.search(tag)
    if not m:
        return None
    val = m.group(1) if m.group(1) is not None else m.group(2)
    return val.strip() if val else None


_TRAILING_BROKEN_TAG_RE = re.compile(r"(?:</?\s*span\b[^>]*|</?)\s*\Z", re.I)


def strip_trailing_broken_spans(text: str) -> str:
    # Remove repeatedly in case there are multiple trailing fragments
    while _TRAILING_BROKEN_TAG_RE.search(text):
        text = _TRAILING_BROKEN_TAG_RE.sub("", text)
    return text


def _spans_to_raw_annotations(text: str) -> List[SpanRecord]:
    """
    Parses <span class="Label"> ... </span> segments and returns:
      - clean_text: input with span tags removed
      - annotations: list of dicts {start, end, label, text}
        indices refer to positions in clean_text
    """
    text = strip_trailing_broken_spans(text)  # <-- pre-pass
    i = 0
    clean = []
    clean_idx = 0
    stack = []
    annotations = []

    n = len(text)
    while i < n:
        m_open = OPEN_TAG_RE.match(text, i)
        if m_open:
            tag = m_open.group(0)
            label = _extract_class_label(tag)
            stack.append({"label": label, "begin": clean_idx})
            i = m_open.end()
            continue

        m_close = CLOSE_TAG_RE.match(text, i)
        if m_close:
            if stack:
                opened = stack.pop()
                if opened["label"]:
                    annotations.append(
                        {
                            "begin": opened["begin"],
                            "end": clean_idx,
                            "label": opened["label"],
                            "text": "".join(clean[opened["begin"] : clean_idx]),
                        }
                    )
            i = m_close.end()
            continue

        clean.append(text[i])
        clean_idx += 1
        i += 1

    # Close any dangling spans
    while stack:
        opened = stack.pop()
        if opened["label"] and opened["begin"] < clean_idx:
            annotations.append(
                {
                    "begin": opened["begin"],
                    "end": clean_idx,
                    "label": opened["label"],
                    "text": "".join(clean[opened["begin"] : clean_idx]),
                }
            )
    annotations = split_label_into_category_subtype(annotations)

    return annotations


def remap_annotations_to_original(alignment, annotations, annotated_text, orig_text):
    """
    alignment.indices is expected to be a (2, N) array-like where:
      - row 0: indices into annotated_text (or -1 for gaps)
      - row 1: indices into orig_text (or -1 for gaps)
    annotations: list of dicts with keys: start, end, label, text
                 [start, end) are offsets in annotated_text
    Returns: list of dicts with original spans: start, end, label, text
    Raises: ValueError on any internal misalignment within an annotation span,
            with detailed, human-friendly diagnostics.
    """
    import numpy as np

    def preview(s, n=80):
        if s is None:
            return "<None>"
        s = s.replace("\n", "\\n")
        return s if len(s) <= n else s[: n - 3] + "..."

    def ctx_slice(s, start, end, pad=20):
        a = max(0, start - pad)
        b = min(len(s), end + pad)
        return s[a:b], a, b

    def nearest_mapping_hint(idx, mapping):
        # Return quick neighbors around idx to show where mapping breaks
        left = next((i for i in range(idx - 1, -1, -1) if i in mapping), None)
        right = next((i for i in range(idx + 1, idx + 50) if i in mapping), None)
        parts = []
        if left is not None:
            parts.append(f"{left}→{mapping[left]}")
        if right is not None:
            parts.append(f"{right}→{mapping[right]}")
        return ", ".join(parts) if parts else "no nearby mapped neighbors"

    def first_diff(a, b):
        m = min(len(a), len(b))
        for i in range(m):
            if a[i] != b[i]:
                return i
        return m if len(a) != len(b) else -1

    a = np.asarray(alignment.indices)
    if a.shape[0] != 2:
        raise ValueError(f"alignment.indices must have shape (2, N); got {a.shape}")

    ann_idx, orig_idx = a[0], a[1]

    # Build mapping from annotated index -> original index
    mapping = {}
    for ai, oi in zip(ann_idx, orig_idx):
        if ai != -1 and oi != -1:
            mapping[int(ai)] = int(oi)

    remapped = []
    for k, ann in enumerate(annotations):
        s, e = int(ann["begin"]), int(ann["end"])  # [s, e)
        label = ann.get("label")
        text = ann.get("text", "")

        # Extra safeguard: ensure the provided 'text' matches annotated_text span
        annotated_slice = annotated_text[s:e]
        if text and annotated_slice != text:
            d = first_diff(annotated_slice, text)
            diff_msg = (
                f" (first difference at +{d}: "
                f"annotated='{annotated_slice[d:d+1] or '∅'}', provided='{text[d:d+1] or '∅'}')"
                if d != -1
                else " (length mismatch)"
            )
            ctx_ann, a0, a1 = ctx_slice(annotated_text, s, e)
            raise ValueError(
                "Provided annotation text doesn't match annotated_text slice.\n"
                f"  label        : {label}\n"
                f"  ann idx      : [{s}:{e}] (len={e - s})\n"
                f"  annotated    : '{preview(annotated_slice)}'\n"
                f"  provided text: '{preview(text)}'{diff_msg}\n"
                f"  context(annotated_text[{a0}:{a1}]): '{preview(ctx_ann)}'"
            )

        chars = list(range(s, e))
        orig_positions = []
        for i in chars:
            if i not in mapping:
                hint = nearest_mapping_hint(i, mapping)
                ch = annotated_text[i : i + 1]
                raise ValueError(
                    "Alignment gap inside annotation span.\n"
                    f"  label            : {label}\n"
                    f"  annotated index  : {i} (char='{preview(ch)}') inside [{s},{e})\n"
                    f"  hint (neighbors) : {hint}\n"
                    "  This likely means the alignment has a gap (-1) or skipped index covering this character."
                )
            orig_positions.append(mapping[i])

        # Ensure contiguity
        for j in range(1, len(orig_positions)):
            if orig_positions[j] != orig_positions[j - 1] + 1:
                i_prev, i_curr = chars[j - 1], chars[j]
                o_prev, o_curr = orig_positions[j - 1], orig_positions[j]
                ch_prev = annotated_text[i_prev : i_prev + 1]
                ch_curr = annotated_text[i_curr : i_curr + 1]
                # Provide local context around the break on the original text
                o_ctx_start = max(0, o_prev - 10)
                o_ctx_end = min(len(orig_text), o_curr + 10)
                o_ctx = orig_text[o_ctx_start:o_ctx_end]
                raise ValueError(
                    "Non-contiguous mapping inside annotation span.\n"
                    f"  label               : {label}\n"
                    f"  annotated jump      : {i_prev}('{preview(ch_prev)}') → {i_curr}('{preview(ch_curr)}')\n"
                    f"  original jump       : {o_prev} → {o_curr} (gap={o_curr - o_prev - 1})\n"
                    f"  annotated span idxs : [{s}:{e}] (len={e - s})\n"
                    f"  original ctx[{o_ctx_start}:{o_ctx_end}]: '{preview(o_ctx)}'\n"
                    "  The original indices should increase by exactly 1 at each annotated step."
                )

        start_orig = orig_positions[0]
        end_orig = orig_positions[-1] + 1

        # Verify strict text match
        orig_slice = orig_text[start_orig:end_orig]
        expect = text if text else annotated_slice
        if orig_slice != expect:
            d = first_diff(orig_slice, expect)
            diff_msg = (
                f" (first difference at +{d}: "
                f"orig='{orig_slice[d:d+1] or '∅'}', expected='{expect[d:d+1] or '∅'}')"
                if d != -1
                else " (length mismatch)"
            )
            ctx_orig, o0, o1 = ctx_slice(orig_text, start_orig, end_orig)
            ctx_ann, a0, a1 = ctx_slice(annotated_text, s, e)
            raise ValueError(
                "Text mismatch after alignment remapping.\n"
                f"  label                   : {label}\n"
                f"  annotated[{s}:{e}]      : '{preview(annotated_slice)}'\n"
                f"  orig[{start_orig}:{end_orig}] : '{preview(orig_slice)}'\n"
                f"  expected (from ann text): '{preview(expect)}'{diff_msg}\n"
                f"  context(orig_text[{o0}:{o1}]): '{preview(ctx_orig)}'\n"
                f"  context(annotated_text[{a0}:{a1}]): '{preview(ctx_ann)}'"
            )

        remapped.append(
            {
                "begin": start_orig,
                "end": end_orig,
                "label": label,
                "text": expect,
            }
        )

    return remapped


def spans_to_annotations(annotated_text: str, original_text: str):
    """
    Convert annotated text with <span class="Label">...</span> tags to
    list of annotations with start/end indices referring to original_text.
    """
    annotations = _spans_to_raw_annotations(annotated_text)
    clean_annotated = remove_html_spans(annotated_text)

    # Only remap if the cleaned annotated text differs from the original text
    if clean_annotated != original_text:
        # Align on the SAME pair you used in the working loop: cleaned vs raw
        alignment = align_texts(clean_annotated, original_text)
        annotations = remap_annotations_to_original(
            alignment, annotations, clean_annotated, original_text
        )

    return annotations


def annotations_to_spans(text, annotations):
    # Sort annotations so that earlier spans are replaced first
    annotations = sorted(annotations, key=lambda x: x["begin"])

    result = ""
    last_idx = 0

    for ann in annotations:
        start, end, label = ann["begin"], ann["end"], ann["label"]

        # Add text before the annotation
        result += text[last_idx:start]

        # Wrap the annotated text
        result += f'<span class="{label}">{text[start:end]}</span>'

        last_idx = end

    # Add the remaining text
    result += text[last_idx:]

    return result


def normalize_to_inception_tokens(
    text: str, spans: List[Dict], lang: str = "nl"
) -> List[Dict]:
    locale = Locale(lang)
    bi = BreakIterator.getWordInstance(locale)
    bi.setText(text)

    # Build token boundaries
    boundaries = []
    start = bi.first()
    end = bi.next()
    while end != BreakIterator.DONE:
        token = text[start:end]
        if token.strip():
            boundaries.append((start, end))
        start = end
        end = bi.next()

    def snap_begin(value: int) -> int:
        return (
            min(boundaries, key=lambda b: abs(value - b[0]))[0] if boundaries else value
        )

    def snap_end(value: int) -> int:
        return (
            min(boundaries, key=lambda b: abs(value - b[1]))[1] if boundaries else value
        )

    # Snap and track original begin for tie-breaking
    candidates = []
    for sp in spans:
        obegin = int(sp["begin"])
        oend = int(sp["end"])
        nbeg = snap_begin(obegin)
        nend = snap_end(oend)

        cand = dict(sp)
        cand["begin"] = nbeg
        cand["end"] = nend
        cand["_orig_begin"] = obegin
        candidates.append(cand)

    # Delegate dedup + containment removal
    return deduplicate_spans(candidates)


def split_label_into_category_subtype(
    span_annotation: List[Dict[str, str | int]],
) -> List[SpanRecord]:
    """
    Convert list of span annotations with 'begin', 'end', 'label', 'text' keys
    to list of tuples (begin, end, label, text) as expected by Inception upload.
    """
    for span in span_annotation:
        category = span["label"].split(":")[0]
        subtype = span["label"].split(":")[1] if ":" in span["label"] else None
        span["Category"] = category

        if subtype:
            span["Subtype"] = subtype

    return span_annotation


# ------- JSON ---------- #
def salvage_json_list(json_str: str) -> str:
    """
    Salvage as many COMPLETE top-level elements from a JSON list as possible.
    Works for lists that contain dicts, lists, strings, numbers, etc.
    Returns a valid JSON string for the salvaged list, or None if nothing usable.
    """
    if not isinstance(json_str, str):
        return None

    # Find the first '[' to allow for prefixes or logs before the JSON
    start = json_str.find("[")
    if start == -1:
        return None

    dec = JSONDecoder()
    i = start + 1
    n = len(json_str)
    elements = []

    def _skip_ws(k: int) -> int:
        while k < n and json_str[k] in " \t\r\n":
            k += 1
        return k

    i = _skip_ws(i)

    while i < n:
        # If the list closes cleanly, we’re done.
        if json_str[i : i + 1] == "]":
            break

        # Try to decode the next complete element.
        try:
            obj, end = dec.raw_decode(json_str, i)
            elements.append(obj)
            i = _skip_ws(end)

            # Expect either ',' (more elements) or ']' (end of list)
            if i < n and json_str[i] == ",":
                i = _skip_ws(i + 1)
                continue
            elif i < n and json_str[i] == "]":
                break
            else:
                # Truncated after a complete element; stop cleanly.
                break

        except json.JSONDecodeError:
            # Incomplete/truncated element; stop without adding it.
            break

    if not elements:
        # If there was clearly a list but no complete elements parsed, return empty list.
        return "[]"

    # Re-serialize the salvaged elements as a valid JSON list.
    return json.dumps(elements, ensure_ascii=False)


def parse_json_list(json_str: str):
    """
    Parse a JSON list; if it fails, salvage as many items as possible and return the list.
    Raises ValueError only if the input does not contain a JSON list at all.
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        raise ValueError("JSON output is not a list")
    except json.JSONDecodeError:
        fixed = salvage_json_list(json_str)
        if fixed is None:
            raise ValueError("Invalid JSON and nothing salvageable")
        return json.loads(fixed)


def _span_from_alignment(al) -> Tuple[int, int]:
    t_blocks = al.aligned[1]
    return t_blocks[0][0], t_blocks[-1][1]


def _similarity(al, query_len: int, match_score: float) -> float:
    return al.score / (match_score * max(1, query_len))


def _choose_mask_char(phrase: str) -> str:
    # pick a char that won't appear in the phrase (so it won't align there again)
    for ch in ("\x00", "\uffff", "\U0010ffff", "\x01"):
        if ch not in phrase:
            return ch
    # worst case: pick a char not in raw later by replacing with a rare symbol
    return "\u25a0"  # ■


def find_all_occurrences(
    raw_text: str,
    phrase: str,
    aligner: Optional[Align.PairwiseAligner] = None,
    min_similarity: float = 0.85,
    case_insensitive: bool = True,
    max_hits: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    if not phrase:
        return []
    if aligner is None:
        aligner = make_aligner()

    raw_work = raw_text.lower() if case_insensitive else raw_text
    phrase_work = phrase.lower() if case_insensitive else phrase
    mask_char = _choose_mask_char(phrase_work)

    hits: List[Tuple[int, int, float]] = []

    while True:
        aligns = aligner.align(phrase_work, raw_work)
        try:
            al = aligns[
                0
            ]  # may raise IndexError if no positive-scoring local alignment
        except IndexError:
            break

        sim = _similarity(al, len(phrase_work), aligner.match_score)
        if sim < min_similarity:
            break

        b, e = _span_from_alignment(al)
        if e <= b:
            break

        hits.append((b, e, sim))

        # mask found region to get next non-overlapping hit
        raw_work = raw_work[:b] + (mask_char * (e - b)) + raw_work[e:]

        if max_hits is not None and len(hits) >= max_hits:
            break

    return hits


def extract_dict_spans(
    raw_text: str,
    annotations: List[Dict[str, str]],
    min_similarity: float = 0.85,
    aligner: Optional[Align.PairwiseAligner] = None,
) -> List[Dict[str, Any]]:
    if aligner is None:
        aligner = make_aligner()

    results: List[Dict[str, Any]] = []
    for ann in annotations:
        phrase = ann.get("annotated_text", "")
        label = ann.get("label", "")
        if not phrase.strip():
            continue

        for b, e, sim in find_all_occurrences(
            raw_text, phrase, aligner, min_similarity
        ):
            b = int(b)
            e = int(e)
            if ":" in label:
                cat, sub = label.split(":", 1)
                results.append(
                    {
                        "begin": b,
                        "end": e,
                        "label": label,
                        "text": raw_text[b:e],
                        "Category": cat,
                        "Subtype": sub,
                        "similarity": sim,
                    }
                )
            else:
                results.append(
                    {
                        "begin": b,
                        "end": e,
                        "label": label,
                        "text": raw_text[b:e],
                        "Category": label,
                        "similarity": sim,
                    }
                )

    results.sort(key=lambda x: (x["begin"], x["end"]))
    return results


def extract_tuple_spans(
    raw_text: str,
    annotations: List[Dict[str, str]],
    min_similarity: float = 0.85,
    aligner: Optional[Align.PairwiseAligner] = None,
) -> List[Dict[str, Any]]:
    if aligner is None:
        aligner = make_aligner()

    results: List[Dict[str, Any]] = []
    for ann in annotations:
        phrase = ann[0]
        label = ann[1]
        if not phrase.strip():
            continue

        for b, e, sim in find_all_occurrences(
            raw_text, phrase, aligner, min_similarity
        ):
            if ":" in label:
                cat, sub = label.split(":", 1)
                results.append(
                    {
                        "begin": b,
                        "end": e,
                        "label": label,
                        "text": raw_text[b:e],
                        "Category": cat,
                        "Subtype": sub,
                        "similarity": sim,
                    }
                )
            else:
                results.append(
                    {
                        "begin": b,
                        "end": e,
                        "label": label,
                        "text": raw_text[b:e],
                        "Category": label,
                        "similarity": sim,
                    }
                )

    results.sort(key=lambda x: (x["begin"], x["end"]))
    return results


# ------- Utilities for trimming/normalizing spans ------- #
def trim_span(s, text):
    b, e = s["begin"], s["end"]
    # move begin right past leading whitespace
    while b < e and text[b].isspace():
        b += 1
    # move end left past trailing whitespace
    while e > b and text[e - 1].isspace():
        e -= 1
    s["begin"], s["end"] = b, e
    s["text"] = text[b:e]
    return s


def normalize_spans(spans, text):
    out = []
    for s in spans:
        s = trim_span(dict(s), text)
        if s["begin"] >= s["end"]:
            # span collapsed to nothing (was only whitespace) — skip it
            continue
        out.append(s)
    return out


def deduplicate_spans(spans: List[Dict]) -> List[Dict]:
    """
    Deduplicate + collapse contained spans.

    - Spans are deduped on (begin, end).
    - If multiple spans have same (begin, end), keep the one with
      lowest `_orig_begin` (or `begin` if not present).
    - If a span is strictly contained by another, keep only the outer span.
    """

    # 1) Dedup by exact (begin, end) with tie-break on _orig_begin
    chosen = {}
    for s in spans:
        b = int(s["begin"])
        e = int(s["end"])
        ob = int(s.get("_orig_begin", b))
        key = (b, e)

        if key not in chosen or ob < int(
            chosen[key].get("_orig_begin", chosen[key]["begin"])
        ):
            chosen[key] = s

    # 2) Sort so outers come before inners
    normalized = sorted(
        chosen.values(),
        key=lambda s: (
            int(s["begin"]),
            -int(s["end"]),  # longer first
            int(s.get("_orig_begin", s["begin"])),
        ),
    )

    # 3) Remove strictly contained spans
    kept = []
    for s in normalized:
        sb, se = int(s["begin"]), int(s["end"])
        contained = any(
            (kb <= sb and ke >= se) and (kb < sb or ke > se)
            for k in kept
            for kb, ke in [(int(k["begin"]), int(k["end"]))]
        )
        if not contained:
            kept.append(s)

    # 4) Optional: final stable sort + cleanup of _orig_begin
    kept = sorted(
        kept,
        key=lambda s: (
            int(s["begin"]),
            int(s["end"]),
            int(s.get("_orig_begin", s["begin"])),
        ),
    )
    for s in kept:
        s.pop("_orig_begin", None)

    return kept


# ------- Conversion re-exports ------- #
#
# Keep conversion logic in span_annotations.conversion,
# while exposing it here for users importing from transform.
from .deduce import (
    DEFAULT_DEDUCE_TO_WORKFLOW_LABEL,
    DEFAULT_WORKFLOW_TO_DEDUCE_TAG,
    deduce_annotation_to_workflow,
    deduce_annotations_to_workflow,
    workflow_annotation_to_deduce,
    workflow_annotations_to_deduce,
    workflow_label_to_deduce_tag,
)
