from __future__ import annotations

from pdfchat.grounding import looks_like_refusal, verify
from pdfchat.models import ScoredChunk
from tests.conftest import FakeChatModel, make_chunk


def sources() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=make_chunk("doc1:00000", "Water boils at 100 degrees Celsius."), score=1.0
        )
    ]


def test_verify_reports_a_grounded_answer():
    model = FakeChatModel(['{"verdict":"grounded","unsupported_claims":[],"note":"All good."}'])
    verdict, note, usage = verify("Water boils at 100C [S1].", sources(), model)
    assert verdict == "grounded"
    assert note == "All good."
    assert usage.calls == 1


def test_verify_surfaces_unsupported_claims():
    model = FakeChatModel(
        [
            '{"verdict":"partially_grounded","unsupported_claims":["it freezes at 0"],"note":"Mostly."}'
        ]
    )
    verdict, note, _ = verify("Water boils at 100C and freezes at 0C [S1].", sources(), model)
    assert verdict == "partially_grounded"
    assert "it freezes at 0" in note


def test_uncited_answers_short_circuit_without_a_model_call():
    model = FakeChatModel(["should not be called"])
    verdict, _note, usage = verify("Water boils at 100 degrees.", sources(), model)
    assert verdict == "ungrounded"
    assert model.prompts == []
    assert usage.calls == 0


def test_refusals_are_treated_as_grounded_without_a_call():
    model = FakeChatModel(["should not be called"])
    verdict, _note, _usage = verify("The documents don't cover this.", sources(), model)
    assert verdict == "grounded"
    assert model.prompts == []


def test_no_sources_means_unchecked():
    verdict, _note, _usage = verify("Anything [S1].", [], FakeChatModel())
    assert verdict == "unchecked"


def test_malformed_verifier_output_degrades_to_unchecked():
    model = FakeChatModel(["not json at all"])
    verdict, _note, _usage = verify("Claim [S1].", sources(), model)
    assert verdict == "unchecked"


def test_unrecognised_verdict_degrades_to_unchecked():
    model = FakeChatModel(['{"verdict":"probably fine"}'])
    verdict, _note, _usage = verify("Claim [S1].", sources(), model)
    assert verdict == "unchecked"


def test_verifier_exception_does_not_propagate():
    class Exploding(FakeChatModel):
        def complete(self, system, user, *, max_tokens=None):
            raise RuntimeError("upstream is down")

    verdict, note, _usage = verify("Claim [S1].", sources(), Exploding())
    assert verdict == "unchecked"
    assert "could not be completed" in note


def test_looks_like_refusal_is_case_insensitive():
    assert looks_like_refusal("The documents don't cover this. Nothing matched.")
    assert not looks_like_refusal("The documents describe photosynthesis.")
