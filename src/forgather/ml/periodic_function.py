from typing import Callable
from .trainer_types import IntervalStrategy


class PeriodicFunction:
    """
    A periodic function caller, which calls 'f' every 'period' steps
    """

    def __init__(
        self,
        strategy: IntervalStrategy,
        period: int,
        epoch_period: int,
        f: Callable,
        first_step=0,
    ):
        self.period = period
        self.f = f
        match strategy:
            case IntervalStrategy.NO:
                self.f = lambda: None
            case IntervalStrategy.STEPS:
                pass
            case IntervalStrategy.EPOCH:
                self.period = epoch_period
            case _:
                pass
        assert self.period > 0
        if first_step >= 0:
            self.counter = first_step % self.period
        else:
            first_step = period - 1

    def reset(self) -> None:
        self.counter = 0

    def step(self, *args, **kwargs) -> None:
        self.counter += 1
        if self.counter == self.period:
            self.f(*args, **kwargs)
            self.reset()

    def count(self) -> int:
        return self.counter
