"""Reusable decorators for tensor validation and manipulation."""

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import torch

__all__ = ["require_dim", "accept_unbatched"]

P = ParamSpec("P")  # preserves parameter types through decorator wrappers
T = TypeVar("T")  # return type of the wrapped function


def require_dim(*allowed_dims: int) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to enforce allowed dimensionalities on all tensor arguments.

    Inspects all positional and keyword arguments. Any `torch.Tensor` argument
    must have an `ndim` present in `allowed_dims`.

    Args:
        *allowed_dims: Acceptable numbers of dimensions.

    Returns:
        A decorator that validates tensor shapes at call time.

    Raises:
        ValueError: If any tensor argument has a dimensionality not in `allowed_dims`.
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for arg in (*args, *kwargs.values()):
                if isinstance(arg, torch.Tensor) and arg.ndim not in allowed_dims:
                    raise ValueError(
                        f"Expected tensor with ndim in {allowed_dims}, "
                        f"got ndim={arg.ndim} (shape {tuple(arg.shape)})."
                    )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def accept_unbatched(dim: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that transparently adds and removes a batch dimension.

    If the primary `data` argument is `dim`-dimensional, it is promoted to `dim+1`
    dimensions before the call and squeezed back afterward. A `dim+1`-dimensional input
    is passed through unchanged.

    Args:
        dim: The dimensionality of an unbatched input tensor.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Method call: first arg is self (not a tensor), second is the tensor.
            if (
                len(args) >= 2
                and not isinstance(args[0], torch.Tensor)
                and isinstance(args[1], torch.Tensor)
            ):
                self_arg, data, rest = args[0], args[1], args[2:]
                promoted = data.ndim == dim
                if promoted:
                    data = data.unsqueeze(0)
                out = fn(self_arg, data, *rest, **kwargs)
                if promoted and isinstance(out, torch.Tensor):
                    return out.squeeze(0)
                return out

            # Standalone: tensor passed as first positional argument.
            if args and isinstance(args[0], torch.Tensor):
                data, rest = args[0], args[1:]
                promoted = data.ndim == dim
                if promoted:
                    data = data.unsqueeze(0)
                out = fn(data, *rest, **kwargs)
                if promoted and isinstance(out, torch.Tensor):
                    return out.squeeze(0)
                return out

            # Standalone: tensor passed as keyword argument named "data".
            if "data" in kwargs and isinstance(kwargs["data"], torch.Tensor):
                data = kwargs.pop("data")
                promoted = data.ndim == dim
                if promoted:
                    data = data.unsqueeze(0)
                out = fn(*args, data=data, **kwargs)
                if promoted and isinstance(out, torch.Tensor):
                    return out.squeeze(0)
                return out

            return fn(*args, **kwargs)

        return wrapper

    return decorator
