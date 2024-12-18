import copy

import matplotlib.pyplot as plt


class Aggregator:
    def __init__(self):
        self.sum = None
        self.count = None
        self.avg = None
        self.reset()
        self.records = []
        self.links = []

    def reset(self):
        self.count, self.avg, self.sum = 0., 0., 0.

    def snapshot(self):
        if self.count > 0: self.records.append(self.avg)

    def snap_and_reset(self):
        self.snapshot()
        self.reset()

    def update(self, val, count=1):
        self.update_links(val, count)
        self.count += count  # For weighing
        self.sum += count * val
        self.avg = self.sum / self.count

    def create_link(self, split: str):
        if split == "child":  # Can only read values from Parent, (best for multi level averaging)
            new_link = Aggregator()
            self.links.append(new_link)
            return new_link
        elif split == "clone":  # Can read and modify only records from parent (best for quick access of average values from anywhere)
            clone = copy.copy(self)
            clone.links = None
            return clone
        elif split == "self":  # Can read and modify anything (Best for quick access)
            return self
        else:
            raise ValueError(f"Invalid split format, expected ['child' or 'clone' or 'self'] got {split}")

    def update_links(self, val, count=1):
        for link in self.links:
            link.count += count  # For weighing
            link.sum += count * val
            link.avg = link.sum / link.count


class Tracker:
    def __init__(self, metrics):
        self.metrics = {metric_name: Aggregator() for metric_name in metrics}
        self.others = {}  # epochs, lr, etc.

    def update(self, metric_val_pairs: dict, count=1):
        for metric_name, val in metric_val_pairs.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].update(val, count)
            else:
                raise KeyError(f"Metric '{metric_name}' not in history.")

    def update_all(self, values, count=1):
        if len(values) != len(self.metrics):
            raise ValueError("Mismatch between number of values and metrics.")
        for metric, val in zip(self.metrics.values(), values):
            metric.update(val, count)

    def create_link(self, metric, split="child"):
        return self.metrics[metric].create_link(split)

    def create_links(self, metrics: list, split="child"):
        links = {}
        for metric in metrics:
            links[metric] = self.create_link(metric, split)
        return links

    def link_all(self, split="child"):
        links = {}
        for metric in self.metrics:
            links[metric] = self.create_link(metric, split)
        return links

    def message(self, prefix=""):
        joiner = " "
        text = f"{prefix} "
        for metric_name, metric in self.metrics.items():
            text += f"{joiner}{metric_name}: {metric.avg:.4f} "
            joiner = ','
        return text.strip()

    def reset_all(self):
        for metric in self.metrics.values():
            metric.reset()

    def snap_and_reset_all(self):
        for metric in self.metrics.values():
            metric.snap_and_reset()

    def __len__(self):
        return len(self.metrics)

    def keys(self):
        return list(self.metrics.keys()) + list(self.others.keys())

    def values(self):
        return [value.records for value in self.metrics.values()] + list(self.others.values())

    def __setitem__(self, key, value):
        if key in self.metrics:
            self.metrics[key].records = value
        else:
            self.others[key] = value

    def __getitem__(self, key):
        if key in self.metrics:
            return self.metrics[key].records
        elif key in self.others:
            return self.others[key]
        else:
            raise KeyError(f"Metric '{key}' not found.")

    def get_history(self):
        return dict(zip(self.keys(), self.values()))

    def plot(self, *metrics, colors=None, line_styles=None, markers=None):
        # Generate a default list of colors if none are provided
        if colors is None:
            # Generate a list of colors using a color map
            colormap = plt.get_cmap('tab10')  # You can choose other colormaps if needed
            colors = [colormap(i / len(metrics)) for i in range(len(metrics))]
        if line_styles is None:
            line_styles = ['-'] * len(metrics)
        if markers is None:
            markers = [''] * len(metrics)

        plt.figure(figsize=(10, 6))

        for idx, metric_name in enumerate(metrics):
            if metric_name not in self.metrics:
                print(f"Warning: Metric '{metric_name}' not found in metrics.")
                continue

            metric = self.metrics[metric_name].records

            epochs = list(range(len(metric)))
            plt.plot(epochs, metric, color=colors[idx], linestyle=line_styles[idx], marker=markers[idx],
                     label=metric_name)

        plt.title('Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

