"""Public errors; provider exceptions remain available as exception causes."""


class TypedEvalsError(Exception):
    """Base error for typed_evals."""


class MissingInputError(TypedEvalsError, ValueError):
    """A metric is missing evidence required to evaluate it."""


class InvalidAnswerError(TypedEvalsError, ValueError):
    """Jev returned an incomplete or inconsistent answer."""


class CalibrationError(TypedEvalsError, ValueError):
    """Calibration data or an artifact is invalid."""


class CalibrationMismatchError(CalibrationError):
    """The model or metric definition differs from the fitted calibrator."""


class DataLeakageError(CalibrationError):
    """Calibration and evaluation data overlap."""
