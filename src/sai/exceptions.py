"""Exceptions raised in SAI context."""


class BaseError(Exception):
    """Base Exception."""


class MissingConfigError(BaseError):
    """Raised when a model does not have a config field."""


class ModelNotFoundError(BaseError):
    """Raised when a model is not registered in the models module."""
