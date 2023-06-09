import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StopTrainingOnSuccessRateThreshold(BaseCallback):
    """
    Stop the training once a threshold in success rate has been reached (i.e. when the
    model is good enough).

    It must be used with the ``EvalCallback``.

    :param success_rate_threshold:  Minimum success rate per episode to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for logging best success rates
    and indicating when training ended because success rate threshold reached.
    """

    def __init__(self, success_rate_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.success_rate_threshold = success_rate_threshold
        self.best_success_rate = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "``StopTrainingOnSuccessRateThreshold`` callback must be used "
            "with an ``EvalCallback``"
        )
        success_rate = np.mean(self.parent._is_success_buffer)
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            print("New best success rate!")
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        continue_training = bool(success_rate < self.success_rate_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the success rate {success_rate:.2f} "
                f" is above the threshold {self.success_rate_threshold}"
            )
        return continue_training
