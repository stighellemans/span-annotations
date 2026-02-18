from span_annotations.alignment import align_texts, make_aligner


def test_make_aligner_defaults():
    aligner = make_aligner()
    assert aligner.mode == "local"
    assert aligner.match_score == 2
    assert aligner.mismatch_score == -2
    assert aligner.open_gap_score == -0.5
    assert aligner.extend_gap_score == -0.1


def test_make_aligner_with_explicit_mode():
    aligner = make_aligner("global")
    assert aligner.mode == "global"


def test_align_texts_returns_best_alignment():
    alignment = align_texts("Patient Jan", "Patient Jan")
    q_blocks, t_blocks = alignment.aligned
    assert q_blocks[0][0] == 0
    assert q_blocks[-1][1] == len("Patient Jan")
    assert t_blocks[0][0] == 0
    assert t_blocks[-1][1] == len("Patient Jan")
