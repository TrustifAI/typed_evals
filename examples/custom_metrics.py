"""Custom metric definitions. Importing this module makes no API calls."""

from typed_evals import Metric

concise = Metric(
    name="concise",
    kind="noul",
    instructions="Does `response` avoid repeating the same substantive point?",
    criteria={
        "true": "Each substantive point is made once, with no redundant restatement.",
        "false": "The response repeats one or more substantive points without adding information.",
    },
    required_fields=("response",),
    pass_definition="No redundant substantive repetition.",
)

reference_alignment = Metric(
    name="reference_alignment",
    kind="choice",
    instructions="How does the answer in `response` compare with the answer in `reference`?",
    criteria={
        "matches": "The response gives the same essential answer as the reference.",
        "contradicts": "The response gives an answer that conflicts with the reference.",
        "undetermined": "The response does not contain enough of an answer to compare.",
    },
    pass_options=("matches",),
    required_fields=("response", "reference"),
    pass_definition="The response gives the same essential answer as the reference.",
)

completeness = Metric(
    name="completeness",
    kind="score",
    instructions="How completely does `response` cover the details requested in `input`?",
    criteria=(
        "None of the requested details are supplied.",
        "Some details are supplied, but essential requested details are missing.",
        "All essential requested details are supplied.",
    ),
    pass_definition="All essential requested details are supplied.",
)
