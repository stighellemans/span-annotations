import span_annotations as pkg


def test_top_level_exports():
    assert "align_texts" in pkg.__all__
    assert "make_aligner" in pkg.__all__
    assert callable(pkg.align_texts)
    assert callable(pkg.make_aligner)
