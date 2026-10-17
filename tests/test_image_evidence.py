import json
from base64 import b64decode

import pytest
from pydantic import ValidationError

from typed_evals import (
    CalibrationExample,
    EvaluationSample,
    ImageInput,
    Metric,
    ToolProposal,
    load_calibration_dataset,
    load_dataset,
)
from typed_evals._utils import digest
from typed_evals.metrics.base import EVALUATION_POLICY


def image(**kwargs):
    return ImageInput.from_bytes(b"image evidence", mime_type="image/png", **kwargs)


def image_metric(**kwargs):
    return Metric(
        name="visual_accuracy",
        instructions="Does the response describe the supplied images accurately?",
        pass_definition="The response matches the visual evidence.",
        required_fields=("input", "response", "images"),
        **kwargs,
    )


def test_minimal_image_dataset_restores_defaults_and_evidence_bytes(tmp_path):
    path = tmp_path / "minimal.json"
    path.write_text(
        '[{"input":"Describe", "response":"A", '
        '"images":[{"data_url":"data:image/png;base64,aW1hZ2UgZXZpZGVuY2U="}]}]'
    )
    sample = load_dataset(path)[0]
    assert sample.images == (image(),)
    assert sample.state(image_metric().required_fields)["images"] == [
        {
            "data_url": "data:image/png;base64,aW1hZ2UgZXZpZGVuY2U=",
            "detail": "auto",
            "label": None,
        }
    ]


@pytest.mark.parametrize(
    "mime_type,detail",
    [
        ("image/png", "auto"),
        ("image/jpeg", "low"),
        ("image/webp", "high"),
        ("image/gif", "original"),
    ],
)
def test_image_bytes_accept_supported_mime_and_detail(mime_type, detail):
    evidence = ImageInput.from_bytes(
        b"opaque image bytes", mime_type=mime_type, detail=detail, label="Reference photo"
    )
    header, encoded = evidence.data_url.split(",", 1)
    assert header == f"data:{mime_type};base64"
    assert b64decode(encoded) == b"opaque image bytes"
    assert evidence.detail == detail
    assert evidence.label == "Reference photo"


@pytest.mark.parametrize(
    "data_url",
    [
        "",
        "https://example.com/photo.png",
        "/tmp/photo.png",
        "data:text/plain;base64,aGVsbG8=",
        "data:image/svg+xml;base64,aGVsbG8=",
        "data:image/jpg;base64,aGVsbG8=",
        "data:image/png,aGVsbG8=",
        "data:image/png;base64,",
        "data:image/png;base64,invalid!",
        "data:image/png;base64,aA",
        "data:image/png;base64,aGVsbG8=\n",
        "data:image/png;base64,☃",
        "data:image/png;base64,aGVsbG8=,extra",
    ],
)
def test_image_input_rejects_remote_paths_unsupported_types_and_invalid_base64(data_url):
    with pytest.raises(ValidationError):
        ImageInput(data_url=data_url)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_url": 123},
        {"detail": "medium"},
        {"detail": None},
        {"label": ""},
        {"label": " \n "},
        {"label": 1},
        {"path": "photo.png"},
    ],
)
def test_image_fields_are_strict(kwargs):
    with pytest.raises(ValidationError):
        ImageInput(**{**image().model_dump(), **kwargs})


def test_image_evidence_and_identity_are_isolated_from_caller_owned_containers():
    evidence = image(label="Reference").model_dump(mode="json")
    images = [evidence]
    sample = EvaluationSample(input="Describe", response="A", images=images)
    original_hash = sample.content_hash

    evidence["data_url"] = ImageInput.from_bytes(b"changed image", mime_type="image/png").data_url
    evidence["label"] = "Changed label"
    images.append(image(label="Unrelated").model_dump(mode="json"))
    exported = sample.state(image_metric().required_fields)
    exported["images"][0]["detail"] = "high"

    assert sample.images == (image(label="Reference"),)
    assert sample.content_hash == original_hash
    assert sample.state(image_metric().required_fields)["images"][0]["detail"] == "auto"


def test_equivalent_base64_encodings_have_one_evidence_identity():
    canonical = ImageInput(data_url="data:image/png;base64,Zg==")
    alias = ImageInput(data_url="data:image/png;base64,Zh==")
    assert alias.data_url == canonical.data_url == "data:image/png;base64,Zg=="
    first = EvaluationSample(input="Describe", response="A", images=(canonical,))
    second = EvaluationSample(input="Describe", response="A", images=(alias,))
    assert first.content_hash == second.content_hash
    fields = image_metric().required_fields
    assert digest(first.state(fields)) == digest(second.state(fields))


def test_image_byte_helpers_reject_empty_data_and_nonbytes():
    with pytest.raises(ValidationError, match="nonempty"):
        ImageInput.from_bytes(b"", mime_type="image/png")
    with pytest.raises(TypeError, match="bytes"):
        ImageInput.from_bytes("text", mime_type="image/png")
    with pytest.raises(ValidationError, match="PNG, JPEG, WebP, or GIF"):
        ImageInput.from_bytes(b"data", mime_type="application/octet-stream")


@pytest.mark.parametrize(
    "suffix,mime_type",
    [
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".WEBP", "image/webp"),
        (".gif", "image/gif"),
    ],
)
def test_image_file_helper_snapshots_bytes(tmp_path, suffix, mime_type):
    path = tmp_path / f"reference{suffix}"
    path.write_bytes(b"first image")
    evidence = ImageInput.from_file(str(path), detail="high", label="Reference")
    path.write_bytes(b"second image")
    path.unlink()
    restored = ImageInput.model_validate_json(evidence.model_dump_json())
    assert restored == ImageInput.from_bytes(
        b"first image", mime_type=mime_type, detail="high", label="Reference"
    )


def test_image_file_helper_rejects_unsupported_suffix_and_missing_file(tmp_path):
    with pytest.raises(ValueError, match="suffix"):
        ImageInput.from_file(tmp_path / "photo.svg")
    with pytest.raises(FileNotFoundError):
        ImageInput.from_file(tmp_path / "missing.png")


@pytest.mark.parametrize("suffix", ["json", "jsonl"])
def test_image_dataset_and_calibration_round_trip(tmp_path, suffix):
    sample = EvaluationSample(
        input="Describe these pictures.",
        response="The images match.",
        images=(image(label="First"), image(detail="high", label="Second")),
    )
    example = CalibrationExample(sample=sample, labels={"visual_accuracy": 1})
    dataset = tmp_path / f"samples.{suffix}"
    calibration = tmp_path / f"calibration.{suffix}"
    data = sample.model_dump(mode="json")
    labeled = example.model_dump(mode="json")
    dataset.write_text(json.dumps([data] if suffix == "json" else data) + "\n")
    calibration.write_text(json.dumps([labeled] if suffix == "json" else labeled) + "\n")
    assert load_dataset(dataset) == [sample]
    assert load_calibration_dataset(calibration) == [example]
    assert load_dataset(dataset)[0].content_hash == sample.content_hash


def test_invalid_image_dataset_reports_the_nested_field(tmp_path):
    dataset = tmp_path / "samples.json"
    dataset.write_text(
        json.dumps(
            [{"input": "Describe", "response": "A", "images": [{"data_url": "secret path"}]}]
        )
    )
    with pytest.raises(ValueError, match=r"images\.0\.data_url") as exc:
        load_dataset(dataset)
    assert "secret path" not in str(exc.value)


@pytest.mark.parametrize("proposal", [None, ToolProposal(name="inspect", arguments={"x": 1})])
def test_text_sample_hash_preserves_existing_identity(proposal):
    sample = EvaluationSample(input="Q", response="A", proposed_tool_call=proposal)
    previous_data = {
        "input": "Q",
        "response": "A",
        "contexts": [],
        "reference": None,
        "trace": [],
        "expected_outcome": None,
    }
    if proposal is not None:
        previous_data["proposed_tool_call"] = proposal.model_dump(mode="json")
    assert sample.content_hash == digest(previous_data)
    assert (
        sample.content_hash
        == EvaluationSample.model_validate(
            {**sample.model_dump(mode="json"), "images": []}
        ).content_hash
    )


def test_authorization_is_validated_and_enters_content_identity():
    base = EvaluationSample(input="Read T-42", response="", contexts=("Tool specification",))
    allowed = base.model_copy(update={"authorization": ("Alice may read T-42",)})
    denied = base.model_copy(update={"authorization": ("Alice may not read T-42",)})
    assert len({base.content_hash, allowed.content_hash, denied.content_hash}) == 3
    assert allowed.state(("authorization",)) == {"authorization": ["Alice may read T-42"]}
    assert "authorization" not in allowed.state(("input", "response", "contexts"))
    with pytest.raises(ValidationError):
        EvaluationSample(input="Read", response="", authorization=("  ",))


def test_image_hash_includes_bytes_order_detail_and_label_but_ignores_metadata():
    evidence = image()
    other = ImageInput.from_bytes(b"different image", mime_type="image/png")
    sample = EvaluationSample(input="Q", response="A", images=(evidence, other))
    changes = [
        (),
        (evidence,),
        (other, evidence),
        (image(detail="high"), other),
        (image(label="Reference"), other),
        (ImageInput.from_bytes(b"image evidence", mime_type="image/jpeg"), other),
    ]
    for images in changes:
        assert sample.content_hash != sample.model_copy(update={"images": images}).content_hash
    assert (
        sample.content_hash
        == sample.model_copy(
            update={"id": "other", "group_id": "group", "metadata": {"label": 1}}
        ).content_hash
    )


def test_image_selection_uses_required_fields_and_reports_missing_evidence():
    metric = image_metric()
    blank = EvaluationSample(input="Describe", response="")
    sample = blank.model_copy(update={"images": (image(),)})
    assert metric.missing_fields(blank) == ["images"]
    assert metric.missing_fields(sample) == []
    assert sample.state(metric.required_fields) == {
        "input": "Describe",
        "response": "",
        "images": [image().model_dump(mode="json")],
    }
    assert sample.state(("input", "response")) == {"input": "Describe", "response": ""}


@pytest.mark.parametrize(
    "required_fields,fingerprint",
    [
        (("input", "response"), "d84ecb496dd2f8b180abaa4311f428d0c499b539befb9e194039140d336e803f"),
        (
            ("input", "response", "proposed_tool_call"),
            "07b4fbf17d8b933ecceafa29617229fb66879d5cc8a5b7ebf26f34562795dfcf",
        ),
        (
            ("input", "response", "images"),
            "0fc165dde28993170ebaca9966a27e4054e95977fac8ec3756e9a8bb77011e54",
        ),
        (
            ("input", "response", "proposed_tool_call", "images"),
            "5648e29ae629247117cafca2042a3e224e24c2c2099e1b0fab5d4d05c20fd09d",
        ),
    ],
)
def test_metric_fingerprints_bind_revised_evidence_policy(required_fields, fingerprint):
    metric = Metric(
        name="accuracy",
        instructions="Does the response match the evidence?",
        pass_definition="The response matches.",
        required_fields=required_fields,
    )
    # The revised evidence policy deliberately invalidates old calibration identities.
    # Recomputing the expected value with digest() would hide serialization regressions.
    assert metric.fingerprint == fingerprint
    assert metric.model_copy(update={"threshold": 0.9}).fingerprint == fingerprint
    expected = {
        "question": metric.instructions,
        "evaluation_policy": EVALUATION_POLICY,
        "allowed_evidence_fields": list(required_fields),
    }
    if "images" in required_fields:
        expected["required_modalities"] = ["text", "image"]
    assert metric.question().instructions == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "kind": "choice",
            "criteria": {"yes": "Accurate", "no": "Inaccurate"},
            "pass_options": ("yes",),
        },
        {"kind": "score", "criteria": ("Inaccurate", "Accurate")},
    ],
)
def test_all_metric_kinds_include_the_image_modality(kwargs):
    assert image_metric(**kwargs).question().instructions["required_modalities"] == [
        "text",
        "image",
    ]
