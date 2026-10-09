"""Image evidence and capability negotiation stay independent of providers."""

import pytest
from conftest import FakeBackend

from typed_evals import (
    AnswerRelevancy,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    ImageInput,
    JevBackend,
    Metric,
    aevaluate,
    evaluate,
)
from typed_evals.errors import MissingInputError, UnsupportedModalityError


def image_metric():
    return Metric(
        name="visual_grounding",
        instructions="Is response supported by the images?",
        required_fields=("input", "response", "images"),
        pass_definition="All claims are supported by the attached images.",
    )


def image_sample():
    return EvaluationSample(
        input="Describe the product.",
        response="The screen is damaged.",
        images=[ImageInput.from_bytes(b"image fixture", mime_type="image/png", label="product")],
        metadata={"private": "not evidence"},
    )


class ImageBackend(FakeBackend):
    # A future Jev or custom adapter opts in through the same optional capability.
    supported_modalities = frozenset({"text", "image"})


@pytest.mark.parametrize("evaluator_type", [Evaluator, EvaluationPipeline])
async def test_provider_neutral_images_reach_an_opted_in_backend(evaluator_type):
    backend = ImageBackend()
    sample = image_sample()
    row = await evaluator_type([image_metric()], backend=backend).aevaluate_one(sample)
    assert row.passed
    assert backend.sessions == backend.closed == 1
    state, questions = backend.calls[0]
    assert state == sample.state({"input", "response", "images"})
    assert "metadata" not in state
    assert questions["visual_grounding"].instructions["required_modalities"] == ["text", "image"]
    assert "data_url" in state["images"][0]
    assert "input_image" not in str(state)
    assert sample.images[0].data_url not in row.model_dump_json()


@pytest.mark.parametrize("errors", ["raise", "record"])
async def test_legacy_backends_reject_images_before_opening_any_batch_session(errors):
    backend = FakeBackend()
    evaluator = Evaluator(
        [AnswerRelevancy(), image_metric()], backend=backend, missing="skip", errors=errors
    )
    with pytest.raises(UnsupportedModalityError, match="does not support image evidence"):
        await evaluator.aevaluate(
            [EvaluationSample(input="text first", response="fine"), image_sample()]
        )
    assert backend.sessions == 0
    assert not backend.calls


async def test_jev_rejects_image_evidence_before_client_use():
    backend = JevBackend(client=object())
    with pytest.raises(UnsupportedModalityError, match="JevBackend"):
        await Evaluator([image_metric()], backend=backend).aevaluate_one(image_sample())
    # The low-level session must also avoid sending native images as ordinary JSON.
    async with backend.session() as session:
        with pytest.raises(UnsupportedModalityError):
            await session.judge(image_sample().state({"images"}), {})


async def test_unused_images_are_filtered_for_existing_text_only_backends():
    backend = FakeBackend()
    row = await Evaluator(backend=backend).aevaluate_one(image_sample())
    assert row.passed
    assert backend.calls[0][0] == {
        "input": "Describe the product.",
        "response": "The screen is damaged.",
    }


@pytest.mark.parametrize("missing", ["raise", "skip"])
async def test_missing_image_evidence_uses_the_existing_missing_input_policy(missing):
    backend = FakeBackend()
    evaluator = Evaluator([image_metric()], backend=backend, missing=missing)
    sample = EvaluationSample(input="Describe", response="")
    if missing == "raise":
        with pytest.raises(MissingInputError, match="images"):
            await evaluator.aevaluate_one(sample)
    else:
        row = await evaluator.aevaluate_one(sample)
        assert row.metrics["visual_grounding"].status == "skipped"
        assert row.passed is None
    assert backend.sessions == 0


@pytest.mark.parametrize("sync", [True, False])
async def test_direct_convenience_api_accepts_typed_and_dictionary_images(sync):
    backend = ImageBackend()
    images = [image_sample().images[0], image_sample().images[0].model_dump(mode="json")]
    options = dict(
        input="Describe both images",
        response="They match",
        images=images,
        metrics=[image_metric()],
        backend=backend,
    )
    row = evaluate(**options) if sync else await aevaluate(**options)
    assert row.passed
    assert len(backend.calls[0][0]["images"]) == 2


async def test_optional_backend_validation_preflights_the_entire_batch():
    class LimitedBackend(ImageBackend):
        def validate_state(self, state):
            if len(state.get("images", ())) > 1:
                raise ValueError("only one image supported")

    backend = LimitedBackend()
    valid = image_sample()
    oversized = valid.model_copy(update={"images": valid.images * 2})
    with pytest.raises(ValueError, match="only one image"):
        await Evaluator([image_metric()], backend=backend).aevaluate([valid, oversized])
    assert backend.sessions == 0
    assert not backend.calls


async def test_backend_validation_hook_must_be_callable():
    class InvalidBackend(ImageBackend):
        validate_state = True

    backend = InvalidBackend()
    with pytest.raises(TypeError, match="validate_state must be callable"):
        await Evaluator([image_metric()], backend=backend).aevaluate_one(image_sample())
    assert backend.sessions == 0


async def test_backend_validation_is_not_called_for_fully_skipped_rows():
    class ValidatingBackend(ImageBackend):
        def validate_state(self, state):
            pytest.fail("Skipped rows have no request to validate")

    backend = ValidatingBackend()
    row = await Evaluator([image_metric()], backend=backend, missing="skip").aevaluate_one(
        EvaluationSample(input="Describe", response="")
    )
    assert row.metrics["visual_grounding"].status == "skipped"
    assert backend.sessions == 0
