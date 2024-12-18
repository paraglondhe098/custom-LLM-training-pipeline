from abc import ABC, abstractmethod
from typing import Optional
import copy


class Callback(ABC):
    """
    Abstract base class for callbacks.
    """

    def __init__(self, pos: int = 1):
        """
        The position at which the callback will run
        Options:
            0 - After training 1 batch (loss.backward() and optimizer.step(), before evaluation/validation (runs for each batch in each epoch) --> return values are not used.
            1 (default) - After validation step. Runs once per epoch. (best for metric tracking, logging, etc.) --> return values (str) are used to print at the end of the epoch
            2 - After training step and before validation step. Runs once per epoch. --> return values are not used.
            Note - for some callbacks, returned value is not used, you can still print within it.
        """
        self.trainer = None
        self.pos = pos

    @abstractmethod
    def event(self) -> Optional[str]:
        """
        Carry out any operation at desired epoch number
        Return:
            (Optional) A message to print in the end of epoch while training
        """
        pass

    # Do not overwrite this method.
    def __call__(self, pos: int) -> Optional[str]:
        try:
            if pos == self.pos:
                return self.event()
            return None
        except Exception as e:
            print(f"Callback Error!: {str(e)}")
            raise e


class IntraEpochReport(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class ImageSaver(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class NotebookLogger(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class TensorBoardLogger(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class CSVLogger(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class LRScheduler(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class SaveCheckpoints(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class EarlyStopping(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class GradientClipping(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class MemoryUsageLogger(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class WeightWatcher(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class ReduceLROnPlateau(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class FeatureMapVisualizer(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class RemoteMonitor(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class NoiseInjector(Callback):
    def __init__(self):
        super().__init__(pos=0)

    def event(self) -> Optional[str]:
        pass


class IntraEpochReport0(Callback):
    def __init__(self, reports_per_epoch: int, report_in_one_line: bool = True):
        """
        Initialize an intra-epoch reporting callback.

        This callback provides periodic performance reporting during training,
        generating a specified number of reports per epoch.

        Args:
            reports_per_epoch (int): Number of reports to generate per epoch.
            report_in_one_line (bool, optional): Whether to display reports
                                                 in a single line. Defaults to True.
        """
        super().__init__(pos=0)  # Runs after each batch

        # Configuration parameters
        self.reports_per_epoch = reports_per_epoch
        self.log_batches = 0

        # Determine message joining strategy
        self.messages_joiner = " " if report_in_one_line else "\n"

    def event(self) -> Optional[str]:
        """
        Generate periodic performance reports during training.

        This method is called after each batch and prints performance
        metrics at specified intervals.

        Returns:
            None: Directly prints the report
        """
        trainer = self.trainer

        # Initialize log batches only in the first epoch
        if trainer.current_epoch == 0:
            # Calculate batch intervals for reporting
            # Ensure at least one report per epoch
            self.log_batches = max(1, trainer.num_batches // self.reports_per_epoch)

        # Check if it's time to log a report
        if trainer.current_batch % self.log_batches == self.log_batches - 1:
            # Construct base message with epoch and batch information
            try:
                # Retrieve the current loss value
                current_loss = trainer.tracker.metrics['loss'].avg

                # Create the base message
                long_message = (f" E-{trainer.current_epoch + 1} "
                                f"batch {trainer.current_batch + 1} "
                                f"loss: {round(current_loss, trainer.roff)}")

                # Add metric information if metrics are available
                if trainer.metrics:
                    for metric in trainer.metrics:
                        # Retrieve the current metric value
                        current_metric = trainer.tracker.metrics[metric].avg

                        # Append metric to the message
                        metric_message = (f"{metric}: "
                                          f"{round(current_metric, trainer.roff)}")

                        # Join messages based on configuration
                        long_message += (self.messages_joiner + metric_message)

                # Print the comprehensive message
                print(long_message)

            except Exception as e:
                # Provide a helpful error message if something goes wrong
                print(f"Error in IntraEpochReport: {e}")


class EarlyStopping0(Callback):
    """
    A sophisticated early stopping mechanism for deep learning model training.

    This callback monitors a specified metric during training and can:
    - Stop training when the metric stops improving
    - Optionally restore the best model runs
    - Handle multiple early stopping instances
    - Provide flexible configuration for minimizing or maximizing metrics

    Args:
        basis (str): The metric to monitor for early stopping (e.g., 'val_loss')
        metric_minimize (bool, optional): Whether to minimize the metric.
            Defaults to True (lower is better, like for loss).
        patience (int, optional): Number of epochs to wait before stopping
            if no improvement. Defaults to 5.
        threshold (float, optional): Minimum threshold for improvement.
            If None, no minimum threshold is applied. Defaults to None.
        restore_best_weights (bool, optional): Whether to restore the model
            runs from the best performing epoch. Defaults to True.
    """
    def __init__(self,
                 basis: str,
                 metric_minimize: bool = True,
                 patience: int = 5,
                 threshold: Optional[float] = None,
                 restore_best_weights: bool = True):
        super().__init__()

        # Tracking variables for metric and improvement
        self.best_epoch = 0
        self.basis = basis
        self.metric_minimize = metric_minimize
        self.patience = patience
        self.threshold = threshold
        self.restore_best_weights = restore_best_weights

        # Initialize best value based on minimization strategy
        self.best_value = float('inf') if metric_minimize else float('-inf')

        # Multi-instance handling
        self.instance = 0
        self.multi_instances = False
        self.called = False

    def event(self) -> Optional[str]:
        """
        Performs early stopping check at each epoch.

        Checks:
        1. Monitors metric improvement
        2. Tracks patience
        3. Handles multiple early stopping callback instances
        4. Optionally restores best model runs

        Returns:
            Optional message about early stopping or best model restoration
        """
        trainer = self.trainer

        # Initialize multi-instance tracking on first epoch
        if trainer.current_epoch == 0:
            self._initialize_multi_instance()

        # Get current metric value from training history
        metric_history = trainer.tracker
        current_metric = metric_history[self.basis][-1]

        # Check if current metric is the best so far
        is_best_metric = (
            (self.metric_minimize and current_metric < self.best_value) or
            (not self.metric_minimize and current_metric > self.best_value)
        )

        if is_best_metric:
            # Update best value and save model runs
            self.best_value = current_metric
            trainer.best_model_weights = copy.deepcopy(trainer.model.state_dict())
            self.best_epoch = trainer.current_epoch
        else:
            # Check threshold and decrement patience
            threshold_met = (
                (self.threshold is None) or
                (self.metric_minimize and self.best_value < self.threshold) or
                (not self.metric_minimize and self.best_value > self.threshold)
            )

            if threshold_met:
                self.patience -= 1

            # Update epoch message with patience info
            trainer.epoch_message += self._get_epoch_message()

        # Check stopping conditions
        last_epoch = (trainer.current_epoch + 1 == trainer.epochs)
        stop_training = (self.patience == 0 or last_epoch)

        if stop_training:
            return self._handle_training_stop(trainer, last_epoch)

        return None

    def _initialize_multi_instance(self):
        """
        Handle multiple early stopping callback instances.
        Ensures unique tracking for each instance.
        """
        trainer = self.trainer
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping) and callback != self:
                self.multi_instances = True
                self.instance = max(callback.instance + 1, self.instance)

        if not self.multi_instances:
            # Clean up unnecessary attributes if single instance
            delattr(self, 'called') if hasattr(self, 'called') else None
            delattr(self, 'instance') if hasattr(self, 'instance') else None

    def _get_epoch_message(self):
        """
        Generate epoch message for tracking patience.
        """
        if self.multi_instances:
            return f" <es{self.instance}-{self.basis}-p-{self.patience}>"
        return f" <es-p-{self.patience}>"

    def _handle_training_stop(self, trainer, last_epoch):
        """
        Handle the training stop process, including:
        - Checking multiple instance conflicts
        - Stopping training
        - Restoring best runs
        - Generating detailed stopping message
        """
        # Check for conflicts in multi-instance scenario
        if self.multi_instances:
            for callback in trainer.callbacks:
                if isinstance(callback, EarlyStopping) and callback != self:
                    if getattr(callback, 'called', False):
                        return None

        # Trigger stopping
        trainer.STOPPER = True

        if last_epoch:
            print(f"Stopping at last epoch {trainer.current_epoch + 1}")
        else:
            print(f"Early-stopping at epoch {trainer.current_epoch + 1}, basis: {self.basis}")

        # Restore best runs if configured
        if self.restore_best_weights:
            return self._generate_restoration_message(trainer)

        return None

    def _generate_restoration_message(self, trainer):
        """
        Generate a detailed message about model restoration and performance.
        """
        history = trainer.tracker
        final_message = (
            "Restoring best runs... " +
            f"{trainer.model.load_state_dict(trainer.best_model_weights)}" +
            f"\n\tBest epoch: {self.best_epoch + 1}," +
            f"\n\tTraining loss: {history['loss'][self.best_epoch]}," +
            f"\n\tValidation loss: {history['val_loss'][self.best_epoch]},"
        )

        # Add metric details if available
        for metric in trainer.metrics:
            final_message += (
                f"\n\tTraining {metric}: {history[metric][self.best_epoch]}," +
                f"\n\tValidation {metric}: {history[f'val_{metric}'][self.best_epoch]}"
            )

        # Mark as called in multi-instance scenario
        if self.multi_instances:
            self.called = True

        return final_message


class LRTracker(Callback):
    def __init__(self):
        super().__init__(pos=1)
        self.trainer.tracker['lr'] = []

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def event(self):
        trainer = self.trainer
        trainer.tracker['lr'].append(self.get_lr(trainer.optimizer))
        return None
