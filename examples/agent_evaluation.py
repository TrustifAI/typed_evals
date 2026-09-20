"""Evaluate an observed failed execution, not an agent's internal reasoning."""

from typed_evals import evaluate


def main():
    result = evaluate(
        preset="agent",
        input="Create a support ticket for the export failure.",
        response="I created ticket T-42 for your export failure.",
        expected_outcome="A new support ticket exists for the reported export failure.",
        trace=[
            {
                "name": "create_ticket",
                "arguments": {"title": "Export failure"},
                "status": "error",
                "output": {"created": False},
                "error": "Permission denied",
            }
        ],
    )
    # Claims of completion are not evidence. Results still depend on Jev's judgment.
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
