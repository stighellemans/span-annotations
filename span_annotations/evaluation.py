from typing import Any, Dict, List, Tuple

import jpype
import jpype.imports
from jpype.types import *

# Start JVM (point to your JDK if not on PATH)
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[])

from java.text import BreakIterator
from java.util import Locale


def segment_tokens(text: str, lang: str = "nl") -> list[str]:
    # Pick locale
    locale = Locale(lang)
    bi = BreakIterator.getWordInstance(locale)
    bi.setText(text)

    tokens = []
    start = bi.first()
    end = bi.next()
    while end != BreakIterator.DONE:
        token = text[start:end].strip()
        if token:
            tokens.append(token)
        start = end
        end = bi.next()
    return tokens


# --- Tokenization with offsets via BreakIterator ---
def tokens_with_offsets(text: str, lang: str = "nl"):
    locale = Locale(lang)
    bi = BreakIterator.getWordInstance(locale)
    bi.setText(text)
    toks = []
    start = bi.first()
    end = bi.next()
    while end != BreakIterator.DONE:
        piece = text[start:end]
        if piece.strip():  # skip pure whitespace
            toks.append((piece, start, end))
        start = end
        end = bi.next()
    return toks  # list[(token, start, end)]


# ---- Span -> token labels (IO: only type, no B/I) ----
def spans_to_io(
    tokens,
    spans,
    label_key="label",
    min_char_overlap=1,  # >=1 char overlap counts
    min_prop_overlap=0.0,  # OR require >= this fraction of token covered (0..1)
):
    n = len(tokens)
    labels = ["O"] * n
    best_olap = [0] * n  # prefer the label with largest overlap per token

    spans_sorted = sorted(spans, key=lambda s: (s["end"] - s["begin"]), reverse=True)

    for sp in spans_sorted:
        sb, se = sp["begin"], sp["end"]
        # choose which field to use as class
        lab = sp.get(label_key) or sp.get("Category") or "ENT"
        # If you store compound labels like "Name:Patient", you can keep them as-is
        # or split by ":" to use only the main type:
        # lab = lab.split(":", 1)[0]

        for i, (_, tb, te) in enumerate(tokens):
            olap = max(0, min(te, se) - max(tb, sb))
            if olap <= 0:
                continue
            tok_len = te - tb
            if olap < min_char_overlap:
                continue
            if tok_len > 0 and (olap / tok_len) < min_prop_overlap:
                continue
            if olap > best_olap[i]:
                labels[i] = lab
                best_olap[i] = olap
    return labels  # list[str], each "O" or "<TYPE>"


# ---- Metrics (IO, entity-type only) ----
def token_level_metrics_io(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    TP = FP = FN = 0
    for gt, pr in zip(y_true, y_pred):
        gt_ent = gt != "O"
        pr_ent = pr != "O"
        if gt_ent and pr_ent:
            if gt == pr:
                TP += 1
            else:
                FP += 1
                FN += 1
        elif not gt_ent and pr_ent:
            FP += 1
        elif gt_ent and not pr_ent:
            FN += 1
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # per-type
    types = sorted({t for t in (y_true + y_pred) if t != "O"})
    per_type = {}
    for typ in types:
        tp = fp = fn = 0
        for gt, pr in zip(y_true, y_pred):
            if pr == typ and gt == typ:
                tp += 1
            elif pr == typ and gt != typ:
                fp += 1
            elif gt == typ and pr != typ:
                fn += 1
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        per_type[typ] = {
            "precision": p,
            "recall": r,
            "f1": (2 * p * r / (p + r) if (p + r) else 0.0),
            "support": sum(1 for t in y_true if t == typ),
        }

    return {
        "micro": {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sum(1 for t in y_true if t != "O"),
        },
        "per_type": per_type,
    }


# --- Helper: group consecutive tokens with same non-O label into spans ---
def _runs(tokens, labels):
    runs = []  # list of dicts: {label, i_start, i_end_incl, char_begin, char_end, text}
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab == "O":
            i += 1
            continue
        j = i
        while j + 1 < len(labels) and labels[j + 1] == lab:
            j += 1
        _, b0, _ = tokens[i]
        _, _, e1 = tokens[j]
        text = "".join(t for (t, _, _) in tokens[i : j + 1])
        # reconstruct with original spacing by slicing from chars:
        char_begin = tokens[i][1]
        char_end = tokens[j][2]
        runs.append(
            {
                "label": lab,
                "i_start": i,
                "i_end": j,
                "char_begin": char_begin,
                "char_end": char_end,
                "text": text,  # raw concat; prefer slice from source if you pass it in
            }
        )
        i = j + 1
    return runs


# --- Verbose confusion at token- and span-level ---
def verbose_report(text, tokens, y_true, y_pred):
    toks = [t for t, _, _ in tokens]

    # ---- TOKEN-LEVEL with explicit buckets ----
    # dict[label] -> {"TP":[], "FP":[], "FP_mislabeled":[], "FN_missed":[], "FN_mislabeled":[]}
    per_type = {}

    def ensure(lbl):
        if lbl not in per_type:
            per_type[lbl] = {
                "TP": [],
                "FP_plain": [],  # predicted lbl but gold is O
                "FP_mislabeled": [],  # predicted lbl but gold is another label
                "FN_missed": [],  # gold lbl but predicted O
                "FN_mislabeled": [],  # gold lbl but predicted another label
            }

    for i, (gt, pr) in enumerate(zip(y_true, y_pred)):
        tok = toks[i]
        gt_ent, pr_ent = (gt != "O"), (pr != "O")

        if gt_ent:
            ensure(gt)
        if pr_ent:
            ensure(pr)

        if gt_ent and pr_ent and gt == pr:
            per_type[gt]["TP"].append((i, tok))
        elif (not gt_ent) and pr_ent:
            # predicted an entity where gold is O -> FP (plain)
            per_type[pr]["FP_plain"].append((i, tok))
        elif gt_ent and (not pr_ent):
            # completely missed a gold entity
            per_type[gt]["FN_missed"].append((i, tok))
        elif gt_ent and pr_ent and gt != pr:
            # mislabel: FP for predicted type, FN for true type (both 'wrong')
            per_type[pr]["FP_mislabeled"].append((i, tok))
            per_type[gt]["FN_mislabeled"].append((i, tok))
        # else: both O -> nothing

    # ---- SPAN-LEVEL with split buckets ----
    gold_runs = _runs(tokens, y_true)
    pred_runs = _runs(tokens, y_pred)

    # helper: any overlap?
    def runs_overlap(a, b):
        return not (a["i_end"] < b["i_start"] or a["i_start"] > b["i_end"])

    span_confusions = {
        "TP": [],
        "FN_missed": [],  # no overlapping predicted span at all
        "FN_mislabeled": [],  # overlapping predicted span exists, but label differs
        "FP_plain": [],  # predicted span where gold is O in that region
        "FP_mislabeled": [],  # predicted span overlaps gold of another label
    }

    matched_pred = set()

    # classify gold spans
    for gr in gold_runs:
        same_label_overlap = False
        diff_label_overlap = False
        for j, pr in enumerate(pred_runs):
            if not runs_overlap(gr, pr):
                continue
            if pr["label"] == gr["label"]:
                same_label_overlap = True
                matched_pred.add(j)
                # we consider TP on first same-label overlap
                break
            else:
                diff_label_overlap = True

        span_text = text[gr["char_begin"] : gr["char_end"]]
        if same_label_overlap:
            span_confusions["TP"].append(
                {
                    "label": gr["label"],
                    "text": span_text,
                    "char_begin": gr["char_begin"],
                    "char_end": gr["char_end"],
                }
            )
        elif diff_label_overlap:
            span_confusions["FN_mislabeled"].append(
                {
                    "label": gr["label"],
                    "text": span_text,
                    "char_begin": gr["char_begin"],
                    "char_end": gr["char_end"],
                }
            )
        else:
            span_confusions["FN_missed"].append(
                {
                    "label": gr["label"],
                    "text": span_text,
                    "char_begin": gr["char_begin"],
                    "char_end": gr["char_end"],
                }
            )

    # classify predicted spans not matched as TP
    for j, pr in enumerate(pred_runs):
        if j in matched_pred:
            continue
        overlaps_gold = False
        overlaps_gold_same = False
        for gr in gold_runs:
            if not runs_overlap(gr, pr):
                continue
            overlaps_gold = True
            if gr["label"] == pr["label"]:
                overlaps_gold_same = True
                break

        span_text = text[pr["char_begin"] : pr["char_end"]]
        if overlaps_gold and not overlaps_gold_same:
            span_confusions["FP_mislabeled"].append(
                {
                    "label": pr["label"],
                    "text": span_text,
                    "char_begin": pr["char_begin"],
                    "char_end": pr["char_end"],
                }
            )
        else:
            # either no gold overlap at all, or (rare) same label overlap not marked TP
            span_confusions["FP_plain"].append(
                {
                    "label": pr["label"],
                    "text": span_text,
                    "char_begin": pr["char_begin"],
                    "char_end": pr["char_end"],
                }
            )

    return {
        "token_level_per_type": per_type,
        "span_level": span_confusions,
    }


def summarize_per_type(per_type):
    """Compact metrics + breakdown per label from the verbose token buckets."""
    out = {}
    for lbl, d in per_type.items():
        tp = len(d["TP"])
        fp = len(d["FP_plain"]) + len(d["FP_mislabeled"])
        fn = len(d["FN_missed"]) + len(d["FN_mislabeled"])
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[lbl] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": tp + fn,
            "breakdown": {
                "fp_plain": len(d["FP_plain"]),
                "fp_mislabeled": len(d["FP_mislabeled"]),
                "fn_missed": len(d["FN_missed"]),
                "fn_mislabeled": len(d["FN_mislabeled"]),
            },
        }
    return out


# --- End-to-end with metrics + verbose ---
def eval_token_level_io(
    text,
    gold_spans,
    pred_spans,
    lang="nl",
    label_key="label",
    min_char_overlap=1,
    min_prop_overlap=0.0,
    verbose=False,
):
    toks = tokens_with_offsets(text, lang)
    y_true = spans_to_io(
        toks, gold_spans, label_key, min_char_overlap, min_prop_overlap
    )
    y_pred = spans_to_io(
        toks, pred_spans, label_key, min_char_overlap, min_prop_overlap
    )

    metrics = token_level_metrics_io(y_true, y_pred)
    result = {
        "tokens": [t for t, _, _ in toks],
        "gold": y_true,
        "pred": y_pred,
        "metrics": metrics,
    }

    if verbose:
        v = verbose_report(text, toks, y_true, y_pred)
        v["token_level_per_type_summary"] = summarize_per_type(
            v["token_level_per_type"]
        )
        result["verbose"] = v

    return result


def _key(span: Dict[str, Any]) -> Tuple[int, int]:
    """Use exact substring (begin, end) as the identity of a span."""
    return (int(span["begin"]), int(span["end"]))


def _label(span: Dict[str, Any], label_key: str = "label") -> str:
    """Pick which field represents the label; defaults to 'label'."""
    return str(span.get(label_key, ""))


def evaluate_span_edits(
    pred_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    *,
    label_key: str = "label",
) -> Dict[str, Any]:
    """
    Compute minimal operations as an annotation-effort evaluation measure:
      1) Addition  (add a new span when no span exists on that exact substring)
      2) Deletion    (remove a span that should not exist)
      3) Edit   (same substring, wrong label)
    Identity of a span is (begin, end). Overlaps don't matter; only exact substring match counts.
    Lower operation counts indicate better annotation quality.
    """
    pred_by_key = {_key(s): s for s in pred_spans}
    gold_by_key = {_key(s): s for s in gold_spans}

    ops: List[Dict[str, Any]] = []
    annotate = delete = adjust = 0

    # 1) For spans that exist in both: either OK or adjust label
    for k in sorted(set(pred_by_key.keys()) & set(gold_by_key.keys())):
        p, g = pred_by_key[k], gold_by_key[k]
        if _label(p, label_key) != _label(g, label_key):
            adjust += 1
            ops.append(
                {
                    "op": "Edit",
                    "begin": k[0],
                    "end": k[1],
                    "from": _label(p, label_key),
                    "to": _label(g, label_key),
                    "text": g.get("text", p.get("text", "")),
                }
            )

    # 2) For gold-only spans: need to annotate
    for k in sorted(set(gold_by_key.keys()) - set(pred_by_key.keys())):
        g = gold_by_key[k]
        annotate += 1
        ops.append(
            {
                "op": "Addition",
                "begin": k[0],
                "end": k[1],
                "label": _label(g, label_key),
                "text": g.get("text", ""),
            }
        )

    # 3) For pred-only spans: need to delete
    for k in sorted(set(pred_by_key.keys()) - set(gold_by_key.keys())):
        p = pred_by_key[k]
        delete += 1
        ops.append(
            {
                "op": "Deletion",
                "begin": k[0],
                "end": k[1],
                "label": _label(p, label_key),
                "text": p.get("text", ""),
            }
        )

    return {
        "counts": {
            "Addition": annotate,
            "Deletion": delete,
            "Edit": adjust,
            "total_ops": annotate + delete + adjust,
        },
        "operations": ops,  # ordered: adjusts, then annotates, then deletes
    }


# --- Example usage ---
if __name__ == "__main__":
    text = "Jan Janssens werd geboren op 15/12/1984. Patiënt: OXN-7091281. Naam: Annelies Vanderheyden (78)."
    gold = [
        {"begin": 0, "end": 12, "label": "Name", "text": "Jan Janssens"},
        {"begin": 35, "end": 45, "label": "Age_Birthdate", "text": "1984"},
        {"begin": 55, "end": 66, "label": "ID", "text": "OXN-7091281"},
        {"begin": 74, "end": 95, "label": "Name", "text": "Annelies Vanderheyden"},
        {"begin": 97, "end": 99, "label": "Age_Birthdate", "text": "78"},
    ]
    pred = [
        {"begin": 0, "end": 3, "label": "Name", "text": "Jan"},
        {"begin": 4, "end": 12, "label": "Name", "text": "Janssens"},
        {"begin": 35, "end": 45, "label": "Name", "text": "1984"},
        {"begin": 55, "end": 66, "label": "ID", "text": "OXN-7091281"},
        {"begin": 74, "end": 95, "label": "Name", "text": "Annelies Vanderheyden"},
        {"begin": 97, "end": 99, "label": "ID", "text": "78"},
    ]

    out = eval_token_level_io(
        text,
        gold,
        pred,
        lang="nl",
        label_key="label",
        min_char_overlap=1,
        min_prop_overlap=0.0,
        verbose=True,
    )

    # Micro metrics
    print("=== Micro metrics ===")
    m = out["metrics"]["micro"]
    print(
        f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  support={m['support']}"
    )

    # Per-type summary with breakdown
    print("\n=== Per-type (token-level) summary ===")
    for lbl, s in out["verbose"]["token_level_per_type_summary"].items():
        b = s["breakdown"]
        print(
            f"[{lbl}] P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} support={s['support']} "
            f"| fp_plain={b['fp_plain']} fp_mislabeled={b['fp_mislabeled']} "
            f"fn_missed={b['fn_missed']} fn_mislabeled={b['fn_mislabeled']}"
        )

    # Span-level buckets
    span = out["verbose"]["span_level"]

    def _count(lst):
        return len(lst)

    print("\n=== Span-level ===")
    print(
        f"TP={_count(span['TP'])}  "
        f"FN_missed={_count(span['FN_missed'])}  FN_mislabeled={_count(span['FN_mislabeled'])}  "
        f"FP_plain={_count(span['FP_plain'])}  FP_mislabeled={_count(span['FP_mislabeled'])}"
    )

    # Example: list Name token FNs (missed & mislabeled separately)
    tl = out["verbose"]["token_level_per_type"]
    if "Name" in tl:
        print("\nName token FNs (missed):", tl["Name"]["FN_missed"])
        print("Name token FNs (mislabeled):", tl["Name"]["FN_mislabeled"])
