from enum import Enum


class Mode(Enum):
    train = 1
    load = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Mode[s]
        except KeyError:
            raise ValueError()
