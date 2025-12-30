# project/imatch/cli_utils.py
"""
Helpers for building CLI argument parsers.
"""

import argparse
from collections.abc import Callable
from typing import Any, Iterable, List, Sequence, TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_T = TypeVar("_T")


def _progress_columns() -> Sequence[Any]:
    """Build a standard colored progress layout shared across scripts."""
    return (
        SpinnerColumn(style="bold blue"),
        TextColumn("[progress.description]{task.description}", style="bold white"),
        BarColumn(
            bar_width=None,
            style="cyan",
            complete_style="green",
            finished_style="green",
            pulse_style="magenta",
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=True),
    )


def create_progress(*, transient: bool = False) -> Progress:
    """
    Return a configured rich.Progress instance with colored bars.

    `transient=True` removes the bar once the task completes.
    """
    return Progress(*_progress_columns(), transient=transient, refresh_per_second=5)


# e.g. bounded_float(low=0.0, high=1.0) ->
def bounded_float(low: float, high: float) -> Callable[[str], float]:
    """
    Return an argparse type validator enforcing low <= value <= high.
    """

    def _validate(raw: str) -> float:
        value = float(raw)
        if not (low <= value <= high):
            raise argparse.ArgumentTypeError(f"value {value} not in [{low}, {high}]")
        return value

    return _validate


def bounded_int(low: int, high: int) -> Callable[[str], int]:
    """
    Return an argparse type validator enforcing low <= value <= high.
    """

    def _validate(raw: str) -> int:
        value = int(raw)
        if not (low <= value <= high):
            raise argparse.ArgumentTypeError(f"value {value} not in [{low}, {high}]")
        return value

    return _validate


def progress_bar(
    run: Callable[..., _T],
    *args: Any,
    description: str | None = None,
    total: int | None = None,
    **kwargs: Any,
) -> _T:
    """
    Execute `run` while rendering a colored progress bar/spinner.

    Use `description` or `total` to override the label or total units.
    """

    task_label = description or getattr(run, "__name__", "Processing")
    with create_progress(transient=True) as progress:
        task_id = progress.add_task(f"[cyan]{task_label}", total=total)
        try:
            result = run(*args, **kwargs)
        finally:
            # Ensure the task visually completes, even when total is unknown.
            if total is None:
                progress.update(task_id, total=1, completed=1)
            else:
                progress.update(task_id, completed=total)
    return result


def token_preview(tokens: Any) -> str:
    def _flatten(seq: Iterable[Any]) -> Iterable[Any]:
        for item in seq:
            if isinstance(item, (list, tuple)):
                yield from _flatten(item)
            else:
                yield item

    raw = tokens.tolist() if hasattr(tokens, "tolist") else tokens
    values: List[Any]
    if isinstance(raw, (list, tuple)):
        values = list(_flatten(raw))
    else:
        values = [raw]
    length = len(values)

    if length == 0:
        return "[]"
    if length >= 6:
        head_count = tail_count = 3
    elif length >= 4:
        head_count = tail_count = 2
    elif length >= 2:
        head_count = tail_count = 1
    else:
        head_count, tail_count = 1, 0

    if head_count + tail_count >= length:
        display_values = values
    else:
        display_values = values[:head_count]
        display_values += ["..."] + values[-tail_count:] if tail_count else ["..."]
    return "[" + ", ".join(str(item) for item in display_values) + "]"
