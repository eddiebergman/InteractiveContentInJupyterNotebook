import numpy as np

def maprange(x, a, b, c, d):
    return (x-a) * (d-c)/(b-a) + c

def round(x, base=1):
    return base * np.round(x/base)

def floor(x, base=1):
    return base * np.floor(x/base)

def ceil(x, base=1):
    return base * np.ceil(x/base)


def map_to_colours(numbers, low, high, palette):
    i = maprange(numbers, low, high, 0, len(palette)-1)
    if type(i) is np.ndarray:
        i = i.astype(int)
    return np.asarray(palette)[i]
