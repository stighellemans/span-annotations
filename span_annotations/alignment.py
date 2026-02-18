from Bio import Align


def make_aligner(type: str = "local") -> Align.PairwiseAligner:
    """Plain global aligner."""
    a = Align.PairwiseAligner()
    a.mode = type
    a.match_score = 2
    a.mismatch_score = -2
    a.open_gap_score = -0.5
    a.extend_gap_score = -0.1
    return a


def align_texts(text1: str, text2: str):
    """Align annotated text with raw text using local alignment (PairwiseAligner)."""
    aligner = make_aligner()

    alignments = aligner.align(text1, text2)
    return alignments[0]
