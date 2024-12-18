import torch
from typing import Optional, Union
from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass


def get_func(metric_name, binary_output):
    mappings = {"accuracy": Accuracy(binary_output),
                "precision": Precision(binary_output),
                "recall": Recall(binary_output),
                "r2score": R2Score()}
    return mappings[metric_name]


class Accuracy(Metric):
    def __init__(self, binary_output=True):
        self.__call = score_funcs.binary_accuracy_score if binary_output else score_funcs.multiclass_accuracy_score

    def __call__(self, y_true, y_pred):
        return self.__call(y_true, y_pred)


class R2Score(Metric):
    def __init__(self):
        self.__call = score_funcs.r2_score

    def __call__(self, y_true, y_pred):
        return self.__call(y_true, y_pred)


class Precision(Metric):
    def __init__(self, binary_output=True, average='macro'):
        self.bin = binary_output
        self.average = average

    def __call__(self, y_true, y_pred):
        return score_funcs.binary_precision_score(y_true,
                                                  y_pred) if self.bin else score_funcs.multiclass_precision_score(
            y_true, y_pred, self.average)


class Recall(Metric):
    def __init__(self, binary_output=True, average='macro'):
        self.bin = binary_output
        self.average = average

    def __call__(self, y_true, y_pred):
        return score_funcs.binary_recall_score(y_true,
                                               y_pred) if self.bin else score_funcs.multiclass_recall_score(
            y_true, y_pred, self.average)


class score_funcs:
    @staticmethod
    def multiclass_accuracy_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
        predictions = torch.argmax(raw_predictions, dim=-1)
        correct = (predictions == labels).sum().item()
        return correct / labels.size(0)

    @staticmethod
    def binary_accuracy_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
        predictions = torch.round(torch.sigmoid(raw_predictions))  # Assuming threshold of 0.5
        correct = (predictions == labels).sum().item()
        return correct / labels.size(0)

    @staticmethod
    def multiclass_precision_score(labels: torch.Tensor, raw_predictions: torch.Tensor,
                                   average: str = 'macro') -> Optional[Union[float, torch.tensor]]:
        y_pred = torch.argmax(raw_predictions, dim=1)
        classes = torch.unique(labels)
        precision_per_class = []

        for cls in classes:
            # True Positives (TP): correctly predicted instances of class `cls`
            tp = torch.sum((labels == cls) & (y_pred == cls)).item()

            # False Positives (FP): instances predicted as class `cls` but are actually some other class
            fp = torch.sum((labels != cls) & (y_pred == cls)).item()

            # Precision calculation for class `cls`
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0  # Modified conditional
            precision_per_class.append(precision)

        precision_per_class = torch.tensor(precision_per_class)

        if average == 'macro':
            return precision_per_class.mean().item()
        elif average == 'micro':
            # Global counts
            tp_total = torch.sum(labels == y_pred).item()
            fp_total = torch.sum(
                (labels != y_pred) & (y_pred != -1)).item()  # Note: Assuming -1 is not a valid class label
            return tp_total / (tp_total + fp_total)
        elif average == 'weighted':
            # Calculate support for each class
            support = torch.tensor([torch.sum(labels == cls).item() for cls in classes])
            return (precision_per_class * support / support.sum()).sum().item()
        else:
            return precision_per_class

    @staticmethod
    def binary_precision_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
        predictions = torch.round(torch.sigmoid(raw_predictions))
        tp = torch.sum((labels == 1) & (predictions == 1)).item()
        fp = torch.sum((labels == 0) & (predictions == 1)).item()
        return tp / (tp + fp) if tp + fp > 0 else 0.0  # Modified conditional

    @staticmethod
    def multiclass_recall_score(labels: torch.Tensor, raw_predictions: torch.Tensor,
                                average: str = 'macro') -> Optional[Union[float, torch.tensor]]:
        # Convert logits to predicted class labels
        predictions = torch.argmax(raw_predictions, dim=1)

        # Get the unique classes
        classes = torch.unique(labels)

        # Initialize variables
        recall_per_class = []

        for cls in classes:
            # True Positives (TP): correctly predicted instances of class `cls`
            tp = torch.sum((labels == cls) & (predictions == cls)).item()

            # False Negatives (FN): instances of class `cls` that were predicted as some other class
            fn = torch.sum((labels == cls) & (predictions != cls)).item()

            # Recall calculation for class `cls`
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0  # Modified conditional

            recall_per_class.append(recall)

        recall_per_class = torch.tensor(recall_per_class)

        if average == 'macro':
            return recall_per_class.mean().item()
        elif average == 'micro':
            # Global counts
            tp_total = torch.sum(labels == predictions).item()
            fn_total = torch.sum((labels != predictions) & (labels == labels)).item()  # Fixed condition
            return tp_total / (tp_total + fn_total)
        elif average == 'weighted':
            # Calculate support for each class
            support = torch.tensor([torch.sum(labels == cls).item() for cls in classes])
            return (recall_per_class * support / support.sum()).sum().item()
        else:
            return recall_per_class

    @staticmethod
    def binary_recall_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
        # Convert logits to predicted class labels (assuming a threshold of 0.5)
        predictions = torch.round(torch.sigmoid(raw_predictions))

        # True Positives (TP): correctly predicted positive instances
        tp = torch.sum((labels == 1) & (predictions == 1)).item()

        # False Negatives (FN): instances of class `1` predicted as `0`
        fn = torch.sum((labels == 1) & (predictions == 0)).item()

        # Recall calculation
        return tp / (tp + fn) if tp + fn > 0 else 0.0  # Modified conditional

    @staticmethod
    def r2_score(labels: torch.Tensor, predictions: torch.Tensor) -> float:
        labels_mean = torch.mean(labels, dim=0)
        tss = torch.sum((labels - labels_mean) ** 2)
        rss = torch.sum((labels - predictions) ** 2)
        delta = 1.e-10
        tss += delta
        return (1 - rss / tss).item()
