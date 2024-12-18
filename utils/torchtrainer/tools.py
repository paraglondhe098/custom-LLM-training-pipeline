from abc import ABC, abstractmethod


class Module(ABC):
    pass


class WeightInitializer(Module):
    def __init__(self):
        pass


class Regularizer(Module):
    def __init__(self):
        pass


class LayerFreezer(Module):
    def __init__(self):
        pass


