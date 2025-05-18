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
        assert period > 0
        self.period = period
        self.counter = first_step + 1
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

    def count(self):
        return self.counter - 1

    def reset(self, value=0) -> None:
        self.counter = value

    def step(self, *args, **kwargs) -> None:
        if self.counter >= 0 and self.counter % self.period == 0:
            self.f(*args, **kwargs)
        self.counter += 1
