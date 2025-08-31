from .trainer_types import IntervalStrategy


class PeriodicFunction:
    def __init__(
        self,
        global_step,
        strategy: IntervalStrategy,
        period: int,
        epoch_period: int,
        first_step=0,
    ):
        assert period > 0
        # Present global step
        self.global_step = global_step

        # First global step where trigger is allowed
        self.first_step = first_step
        # Relative step, since last reset
        self.rel_step = 0
        self.enabled = True
        # Total number of times triggered

        match strategy:
            case IntervalStrategy.NO:
                self.enabled = False
            case IntervalStrategy.STEPS:
                assert period > 0
                self.period = period
            case IntervalStrategy.EPOCH:
                assert epoch_period > 0
                self.period = epoch_period
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

    def __str__(self):
        return f"PeriodicFunction: period={self.period}, first_step={self.first_step}, rel_step={self.rel_step}, global_step={self.global_step}, enabled={self.enabled}"

    def reset(self):
        """
        Reset counter and return relative steps since last reset
        """
        step = self.rel_step
        self.rel_step = 0
        return step

    def step(self, *args, **kwargs) -> int:
        self.global_step += 1
        self.rel_step += 1

        if (
            not self.enabled
            or self.global_step < self.first_step
            or self.rel_step < self.period
        ):
            return False
        return True
