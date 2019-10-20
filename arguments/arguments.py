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


class NumberChannels(Enum):
    three = 3    # RGB
    four = 4     # RGB + NI
    six = 6      # RGB + NI + NDWI + NDVI
    seven = 7    # RGB + NI + NDWI + NDVI + Elevation
    eight = 8    # RGB + NI + NDWI + NDVI + Elevation + Imperviousness

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return NumberChannels[s]
        except KeyError:
            raise ValueError()
