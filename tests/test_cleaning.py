from __future__ import annotations

from pdfchat.cleaning import (
    clean_document_pages,
    clean_page,
    dehyphenate,
    find_repeated_lines,
    looks_like_heading,
    normalize_unicode,
    strip_page_furniture,
    tidy_whitespace,
    unwrap_soft_linebreaks,
)


def test_normalize_unicode_expands_ligatures_and_quotes():
    assert normalize_unicode("ﬁrst ofﬁce") == "first office"
    assert normalize_unicode("“quoted”") == '"quoted"'
    assert normalize_unicode("soft­hyphen") == "softhyphen"


def test_normalize_unicode_strips_control_characters():
    assert normalize_unicode("clean\x00\x07text") == "cleantext"


def test_dehyphenate_rejoins_split_words():
    assert dehyphenate("photosyn-\nthesis") == "photosynthesis"


def test_dehyphenate_keeps_genuine_compounds():
    # Next line starts uppercase, so the hyphen is real punctuation.
    assert dehyphenate("well-\nKnown") == "well-\nKnown"


def test_unwrap_soft_linebreaks_joins_mid_sentence_wraps():
    text = "The cell uses energy\nto move ions across\nthe membrane."
    assert unwrap_soft_linebreaks(text) == "The cell uses energy to move ions across the membrane."


def test_unwrap_soft_linebreaks_preserves_list_structure():
    text = "Steps:\n- first item\n- second item"
    assert unwrap_soft_linebreaks(text).count("\n") == 2


def test_strip_page_furniture_removes_page_numbers():
    text = "Real content here.\n12\nMore content.\nPage 3 of 40\n- 7 -"
    cleaned = strip_page_furniture(text)
    assert "Real content here." in cleaned
    assert "More content." in cleaned
    assert "12" not in cleaned.split("\n")
    assert "Page 3 of 40" not in cleaned


def test_strip_page_furniture_removes_toc_leader_dots():
    assert "...." not in strip_page_furniture("Chapter One........ 5")


def test_find_repeated_lines_detects_running_headers():
    pages = [f"ACME Report 2024\nBody text for page {n}.\nConfidential" for n in range(1, 8)]
    repeated = find_repeated_lines(pages)
    assert "ACME Report #" in repeated
    assert "Confidential" in repeated


def test_find_repeated_lines_ignores_short_documents():
    assert find_repeated_lines(["Header\nbody", "Header\nbody"]) == set()


def test_find_repeated_lines_ignores_body_text():
    # A recurring sentence in the middle of the page is content, not furniture.
    pages = ["Top\n" + "filler\n" * 8 + "the same sentence\n" + "filler\n" * 8 + "Bottom"] * 6
    assert "the same sentence" not in find_repeated_lines(pages)


def test_tidy_whitespace_collapses_runs_and_fixes_punctuation():
    assert tidy_whitespace("word    spaced ,  here .") == "word spaced, here."
    assert tidy_whitespace("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_page_composes_the_whole_pipeline():
    raw = "ACME | 4\nThe mitochon-\ndria produces\nATP eﬃciently.\n4"
    cleaned = clean_page(raw, {"ACME | #"})
    assert cleaned == "The mitochondria produces ATP efficiently."


def test_clean_document_pages_strips_furniture_across_pages():
    pages = [f"Handbook v2\nContent on page {n}.\n{n}" for n in range(1, 6)]
    cleaned = clean_document_pages(pages)
    assert len(cleaned) == 5
    assert all("Handbook" not in page.text for page in cleaned)
    assert cleaned[0].text == "Content on page 1."


def test_clean_document_pages_handles_empty_input():
    assert [p.text for p in clean_document_pages(["", ""])] == ["", ""]


def test_looks_like_heading_recognises_headings_not_prose():
    assert looks_like_heading("CHAPTER THREE")
    assert looks_like_heading("3.1 Photosynthesis Overview")
    assert looks_like_heading("Cellular Respiration Basics")
    assert not looks_like_heading("The cell uses glucose to make energy for the organism.")
    assert not looks_like_heading("This sentence ends with a period.")
