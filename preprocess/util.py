import math
import numpy as np

def is_nan(value):
    if type(value) == int or type(value) == float:
        return math.isnan(value)
    return False

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points
