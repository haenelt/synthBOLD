"""Reusable decorators for tensor validation and manipulation."""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import torch

__all__ = ["log_call", "to_device"]

P = ParamSpec("P")  # preserves parameter types through decorator wrappers
T = TypeVar("T")  # return type of the wrapped function


def to_device[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Decorator that moves every `torch.Tensor` argument to `self.device` before
    calling the wrapped bound method.

    Args:
        fn: Bound method taking `self` followed by any mix of tensor and
            non-tensor positional/keyword arguments.

    Returns:
        Wrapped method that transfers all tensor arguments to `self.device` before
        delegating to `fn`.
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        self, *rest = args
        device = self.device  # type: ignore[attr-defined]
        rest = [a.to(device) if isinstance(a, torch.Tensor) else a for a in rest]
        kwargs = {  # type: ignore[assignment]
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return fn(self, *rest, **kwargs)  # type: ignore[arg-type]

    return wrapper


def log_call[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Decorator that logs method name, input/output tensor shapes, and elapsed time
    at INFO level.

    Logs to a logger named after the instance's module (`type(self).__module__`), so
    output is attributed to the concrete subclass regardless of where the decorated
    method is defined. If the first positional argument (after `self`) is a tensor,
    its shape is logged as the input shape; the return value's shape is logged if it
    is a tensor.

    Args:
        fn: Bound method to wrap, e.g. a `Transform`/`Model`/`BaseGeometry`
            `__call__`.

    Returns:
        Wrapped method that logs around the call to `fn`.
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        self, *rest = args
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start

        logger = logging.getLogger(type(self).__module__)
        input_shape = (
            tuple(rest[0].shape) if rest and isinstance(rest[0], torch.Tensor) else None
        )
        output_shape = tuple(result.shape) if isinstance(result, torch.Tensor) else None
        logger.info(
            "%s.%s input=%s output=%s elapsed=%.4fs",
            type(self).__name__,
            fn.__name__,
            input_shape,
            output_shape,
            elapsed,
        )
        return result

    return wrapper
