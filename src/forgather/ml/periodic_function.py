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
        phase=0,
    ):
        assert period > 0
        self.period = period
        self.phase = phase
        self.counter = 0
        self.enabled = True
        match strategy:
            case IntervalStrategy.NO:
                self.enabled = False
            case IntervalStrategy.STEPS:
                pass
            case IntervalStrategy.EPOCH:
                self.period = epoch_period
            case _:
                pass

    def count(self):
        return self.counter % self.period

    def step(self, *args, **kwargs) -> None:
        self.counter += 1
        if self.enabled and (self.counter + self.phase) % self.period == 0:
            return True
        else:
            return False
