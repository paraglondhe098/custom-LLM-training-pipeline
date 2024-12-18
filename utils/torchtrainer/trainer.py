import time
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from typing import List, Callable, Tuple
from utils.torchtrainer.metrics import *
from utils.torchtrainer.utils import *
from utils.torchtrainer.callbacks import Callback
import torch
from torchsummary import summary
from tqdm import tqdm


class Trainer:
    def __init__(self, model: torch.nn.Module,
                 epochs: int,
                 criterion: torch.nn.Module,
                 input_shape: Tuple[int, ...],
                 output_shape: Tuple[int, ...],
                 optimizer: torch.optim.Optimizer,
                 metrics: Optional[Union[str, List[str]]] = None,
                 callbacks: Optional[List[Callback]] = None,
                 display_time_elapsed: bool = False,
                 roff: int = 5,
                 report_in_one_line: bool = True,
                 clear_cuda_cache: bool = True,
                 use_amp: bool = True,
                 device: torch.device = torch.device('cpu')):

        self.messages_joiner = "  ||  " if report_in_one_line else "\n"
        self.epoch_message = None

        self.num_batches = None
        self.batch_size = None
        self.epochs = epochs
        self.current_epoch = 0
        self.current_batch = 0

        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.metric_fns = {metric_name: get_func(metric_name, binary_output=(output_shape[-1] == 1)) for metric_name in
                           self.metrics}

        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.clear_cuda_cache = clear_cuda_cache
        self.scaler = GradScaler() if (self.device.type == 'cuda' and use_amp) else None
        self.tracker = self.init_tracker()

        self.roff = roff
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.display_time_elapsed = display_time_elapsed

        self.STOPPER = False
        self.external_events = []
        self.best_model_weights = None
        # self.__validate_output()

        callbacks = callbacks if isinstance(callbacks, list) else []
        self.callbacks = []
        for cb in callbacks:
            if isinstance(cb, Callback):
                cb.trainer = self
                self.callbacks.append(cb)
            else:
                raise TypeError("callbacks should be inherited from Callback class (from torchtrainer.callbacks)")

    def init_tracker(self):
        temp = self.metrics + ["loss"]
        metrics = []
        for metric in temp:
            metrics.append(metric)
            metrics.append(f"val_{metric}")
        return Tracker(metrics)

    # @torch.no_grad()
    # def __validate_output(self):
    #     try:
    #         input_shape = [10] + list(self.input_shape)
    #         fake_input = torch.randn(size=input_shape)
    #         fake_output = self.model(fake_input)
    #         if tuple(fake_output.shape[1:]) == self.output_shape:
    #             print("Model is initialized properly")
    #         else:
    #             print(
    #                 f"Given output shape {tuple(fake_output.shape[1:])} and obtained output shape {self.output_shape} did not match.")
    #     except Exception as e:
    #         print("Model is not built/initialized properly")
    #         raise e

    def model_summary(self):
        summary(self.model, self.input_shape)

    def add_metric(self, metric: str, metric_fn: Callable) -> None:
        self.metrics.append(metric)
        self.metric_fns[metric] = metric_fn
        return None

    def add_callback(self, callback: Callback) -> None:
        """
        Adds a callback to the Trainer.

        Note:
            If you're adding a custom callback function, make sure it's inherited
            from the `Callback` abstract base class and overwrites the `run` method,
            otherwise the callback will not run!

        Args:
            callback (Callback): Callback object to add. Must be an instance of
                                 a class inherited from the `Callback` base class.

        """
        if (callback not in self.callbacks) and isinstance(callback, Callback):
            callback.trainer = self
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callback) -> None:
        """
        Removes a callback from the Trainer.

        Args:
            callback: Callback object to remove.
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def __run_callbacks(self, pos: int) -> List[Optional[str]]:
        responses = [callback(pos) for callback in self.callbacks]
        return [response for response in responses if response]

    def __train_fn(self, train_loader: torch.utils.data.DataLoader) -> None:

        # Set to training mode
        self.model.train()
        for self.current_batch, (inputs, labels) in tqdm(enumerate(train_loader), self.epoch_message):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # One Batch Training
            self.optimizer.zero_grad()
            if self.scaler:
                # Mixed precision training
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Normal training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.tracker.update({"loss": loss.item()})
            with torch.no_grad():
                self.tracker.update({metric: self.metric_fns[metric](labels, outputs) for metric in self.metrics})
            self.__run_callbacks(pos=0)

    @torch.no_grad()
    def __validation_fn(self, val_loader: torch.utils.data.DataLoader) -> None:
        # Set to the evaluation mode
        self.model.eval()

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.scaler:
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    val_loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                val_loss = self.criterion(outputs, labels)

            self.tracker.update({"val_loss": val_loss.item()})
            self.tracker.update(
                {"val_" + metric: self.metric_fns[metric](labels, outputs) for metric in self.metrics})

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        """
        Trains the model for the specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.

        Returns:
            None
        """
        self.num_batches = len(train_loader)
        self.batch_size = train_loader.batch_size
        on_gpu = True if self.device.type == 'cuda' else False
        tracker = self.tracker
        # The main Training loop

        start_time = time.time()
        for epoch in range(self.epochs):

            if on_gpu and self.clear_cuda_cache:
                torch.cuda.empty_cache()

            self.epoch_message = f"EPOCH {self.current_epoch + 1}: "

            # Train model
            self.__train_fn(train_loader)
            self.__run_callbacks(pos=2)
            self.__validation_fn(val_loader)
            epoch_statistics = tracker.message("--> Metrics: ")
            tracker.snap_and_reset_all()

            # Run callbacks
            responses = self.__run_callbacks(pos=1)
            print(epoch_statistics)

            if self.display_time_elapsed:
                end_time = time.time()
                print(f"Time elapsed: {end_time - start_time} s")

            for response in responses:
                print(response)

            self.current_epoch += 1

            if self.STOPPER:
                break
        if self.best_model_weights is None:
            self.best_model_weights = copy.deepcopy(self.model.state_dict())
        return tracker.get_history()

    @torch.no_grad()
    def predict(self, data):
        data = data.to(self.device)
        return self.model(data)

    def add_event(self, pos: int):
        """
        Write a custom callback event without explicitly creating new callback class.
        """

        def decorator(event: Callable) -> Optional[Callable]:
            if event.__name__ in self.external_events:
                return
            ct = CallbackTemplate(pos)
            self.external_events.append(event.__name__)
            ct.fn = event
            self.add_callback(ct)
            return event

        return decorator


class CallbackTemplate(Callback):
    def __init__(self, pos):
        self.fn = lambda x=None: None
        super().__init__(pos=pos)

    def event(self) -> Optional[str]:
        self.fn()
        return None
