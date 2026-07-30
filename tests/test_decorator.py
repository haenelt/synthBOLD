import logging
import re
from typing import Any

import pytest
import torch

from synthbold.decorator import log_call, to_device


class DummyDeviceTransform:
    def __init__(self, device: str) -> None:
        self.device = device

    @to_device
    def forward(self, data: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return data * scale


class DummyDeviceTransformExtraArg:
    def __init__(self, device: str) -> None:
        self.device = device

    @to_device
    def forward(self, data: torch.Tensor, factor: float) -> torch.Tensor:
        return data * factor


class DummyDeviceTransformMultiTensor:
    def __init__(self, device: str) -> None:
        self.device = device

    @to_device
    def forward(self, data: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return data + other


class DummyLoggedTransform:
    @log_call
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2


class DummyLoggedNonTensor:
    @log_call
    def forward(self, value: int) -> int:
        return value + 1


class BaseLogged:
    @log_call
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data


class SubLogged(BaseLogged):
    pass


SubLogged.__module__ = "fake.subclass.module"


# --- to_device ---


def test_to_device_calls_tensor_to_with_self_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = DummyDeviceTransform(device="cpu")
    x = torch.ones(2, 3)
    calls: list[tuple[Any, ...]] = []
    original_to = torch.Tensor.to

    def spy_to(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(args)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    t.forward(x)
    assert calls == [("cpu",)]


def test_to_device_result_reflects_moved_tensor() -> None:
    t = DummyDeviceTransform(device="cpu")
    x = torch.ones(2, 3)
    out = t.forward(x, scale=3.0)
    assert torch.all(out == 3.0)


def test_to_device_passes_through_keyword_arguments() -> None:
    t = DummyDeviceTransform(device="cpu")
    x = torch.ones(2, 3)
    out = t.forward(x, scale=5.0)
    assert torch.all(out == 5.0)


def test_to_device_passes_through_extra_positional_arguments() -> None:
    t = DummyDeviceTransformExtraArg(device="cpu")
    x = torch.ones(2, 3)
    out = t.forward(x, 2.0)
    assert torch.all(out == 2.0)


def test_to_device_moves_all_positional_tensor_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = DummyDeviceTransformMultiTensor(device="cpu")
    x = torch.ones(2, 3)
    y = torch.ones(2, 3)
    calls: list[tuple[Any, ...]] = []
    original_to = torch.Tensor.to

    def spy_to(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(args)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    t.forward(x, y)
    assert calls == [("cpu",), ("cpu",)]


def test_to_device_moves_keyword_tensor_argument() -> None:
    t = DummyDeviceTransform(device="cpu")
    x = torch.ones(2, 3)
    out = t.forward(data=x, scale=2.0)
    assert torch.all(out == 2.0)


def test_to_device_preserves_function_metadata() -> None:
    assert DummyDeviceTransform.forward.__name__ == "forward"


# --- log_call ---


def test_log_call_returns_wrapped_result() -> None:
    t = DummyLoggedTransform()
    x = torch.ones(2, 3)
    out = t.forward(x)
    assert torch.all(out == 2.0)


def test_log_call_logs_at_info_level(caplog: pytest.LogCaptureFixture) -> None:
    t = DummyLoggedTransform()
    x = torch.ones(2, 3)
    with caplog.at_level(logging.INFO, logger=DummyLoggedTransform.__module__):
        t.forward(x)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert caplog.records[0].name == DummyLoggedTransform.__module__


def test_log_call_message_contains_class_and_method_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = DummyLoggedTransform()
    x = torch.ones(2, 3)
    with caplog.at_level(logging.INFO, logger=DummyLoggedTransform.__module__):
        t.forward(x)
    message = caplog.records[0].getMessage()
    assert "DummyLoggedTransform.forward" in message


def test_log_call_logs_input_and_output_shapes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = DummyLoggedTransform()
    x = torch.ones(2, 3)
    with caplog.at_level(logging.INFO, logger=DummyLoggedTransform.__module__):
        t.forward(x)
    message = caplog.records[0].getMessage()
    assert "input=(2, 3)" in message
    assert "output=(2, 3)" in message


def test_log_call_input_shape_none_for_non_tensor_arg(
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = DummyLoggedNonTensor()
    with caplog.at_level(logging.INFO, logger=DummyLoggedNonTensor.__module__):
        t.forward(5)
    message = caplog.records[0].getMessage()
    assert "input=None" in message


def test_log_call_output_shape_none_for_non_tensor_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = DummyLoggedNonTensor()
    with caplog.at_level(logging.INFO, logger=DummyLoggedNonTensor.__module__):
        t.forward(5)
    message = caplog.records[0].getMessage()
    assert "output=None" in message


def test_log_call_logs_elapsed_time(caplog: pytest.LogCaptureFixture) -> None:
    t = DummyLoggedTransform()
    x = torch.ones(2, 3)
    with caplog.at_level(logging.INFO, logger=DummyLoggedTransform.__module__):
        t.forward(x)
    message = caplog.records[0].getMessage()
    match = re.search(r"elapsed=([\d.]+)s", message)
    assert match is not None
    assert float(match.group(1)) >= 0.0


def test_log_call_uses_instance_module_not_definition_module(
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = SubLogged()
    x = torch.ones(2, 3)
    with caplog.at_level(logging.INFO, logger="fake.subclass.module"):
        t.forward(x)
    assert caplog.records[0].name == "fake.subclass.module"


def test_log_call_preserves_function_metadata() -> None:
    assert DummyLoggedTransform.forward.__name__ == "forward"
