"""Public errors; transport messages withhold provider contents."""


class TypedEvalsError(Exception):
    """Base error for typed_evals."""


class MissingInputError(TypedEvalsError, ValueError):
    """A metric is missing evidence required to evaluate it."""


class UnsupportedModalityError(TypedEvalsError, ValueError):
    """A backend cannot evaluate a selected evidence modality."""


class InvalidAnswerError(TypedEvalsError, ValueError):
    """A judge returned an incomplete or inconsistent answer."""


class DecisionRefusalError(InvalidAnswerError):
    """The Decisions provider declined an independently named judgment."""


class OpenAIDecisionsError(TypedEvalsError):
    """A Decisions transport failure, with provider response contents withheld."""


class SystemOneError(TypedEvalsError):
    """A System One transport failure, with provider response contents withheld."""


class CalibrationError(TypedEvalsError, ValueError):
    """Calibration data or an artifact is invalid."""


class CalibrationMismatchError(CalibrationError):
    """The model or metric definition differs from the fitted calibrator."""


class DataLeakageError(CalibrationError):
    """Calibration and evaluation data overlap."""
