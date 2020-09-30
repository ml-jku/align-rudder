from abc import ABC
from abc import abstractmethod

from collections import namedtuple
import numpy as np

from dataclasses import dataclass
from dataclasses import field

from collections import namedtuple

VonNeumannMotion = namedtuple('VonNeumannMotion',
                              ['north', 'south', 'west', 'east'],
                              defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])

@dataclass
class Object:
    r"""Defines an object with some of its properties.

    An object can be an obstacle, free space or food etc. It can also have properties like impassable, positions.

    """
    name: str
    value: int
    rgb: tuple
    impassable: bool
    positions: list = field(default_factory=list)


class BaseMaze(ABC):
    def __init__(self, **kwargs):
        objects = self.make_objects()
        assert all([isinstance(obj, Object) for obj in objects])
        self.objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def size(self):
        r"""Returns a pair of (height, width). """
        pass

    @abstractmethod
    def make_objects(self):
        r"""Returns a list of defined objects. """
        pass

    def _convert(self, x, name):
        for obj in self.objects:
            pos = np.asarray(obj.positions)
            if pos.size > 0:
                x[pos[:, 0], pos[:, 1], pos[:, 2]] = getattr(obj, name, None)
        return x

    def to_name(self):
        x = np.empty(self.size, dtype=object)
        return self._convert(x, 'name')

    def to_value(self):
        x = np.empty(self.size, dtype=int)
        return self._convert(x, 'value')

    def to_rgb(self):
        x = np.empty((*self.size, 3), dtype=np.uint8)
        return self._convert(x, 'rgb')

    def to_impassable(self):
        x = np.empty(self.size, dtype=bool)
        return self._convert(x, 'impassable')

    def __repr__(self):
        return f'{self.__class__.__name__}{self.size}'
