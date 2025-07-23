from numba import jit, int16
import datashader as ds
import pandas as pd
import numpy as np
from datashader import transfer_functions as tf
from collections import Counter
import itertools
import PIL

RESOLUTION_X = 1950
RESOLUTION_Y = 1920
MAX_ITERATIONS = 1000

@jit
def periodicity_to_start_gradient(x_init, y_init):
    x = x_init
    y = y_init
    for i in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % RESOLUTION_X, (x + y) % RESOLUTION_Y
        if x == x_init and y == y_init:
            return i
    return -1

@jit
def periodicity_to_start(x_init, y_init):
    x = x_init
    y = y_init
    for i in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % RESOLUTION_X, (x + y) % RESOLUTION_Y
        if x == x_init and y == y_init:
            return 1
    return -1

@jit
def periodicity(x, y):
    visited = [[x, y]]
    for i in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % RESOLUTION_X, (x + y) % RESOLUTION_Y
        if [x, y] in visited:
            return i - visited.index([x, y])
        visited.append([x,y])
    return -1


@jit
def apply_over_matrix(f):
    matrix = []
    for i in range(RESOLUTION_X):
        row = []
        for j in range(RESOLUTION_Y):
            row.append(f(i, j))
        matrix.append(row)
    return matrix

def pixel_movement_matrix(x, y):
    matrix = np.zeros((RESOLUTION_X, RESOLUTION_Y), int)
    for _ in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % RESOLUTION_X, (x + y) % RESOLUTION_Y
        matrix[x][y] += 1
    return matrix


def main():
    #matrix = pixel_movement(2,1)
    matrix = apply_over_matrix(periodicity_to_start)
    print(dict(Counter(itertools.chain.from_iterable(matrix))))

    matrix = np.asarray(matrix)

    n, m = matrix.shape

    x_coords, y_coords = np.mgrid[0:n, 0:m]
    df = pd.DataFrame({
        'x': x_coords.ravel(),
        'y': y_coords.ravel(),
        'value': matrix.ravel()
    })

    canvas = ds.Canvas(plot_width=RESOLUTION_X, plot_height=RESOLUTION_Y)
    agg = canvas.points(df, 'x', 'y', ds.mean('value'))
    img = tf.shade(agg, cmap=["black", "white"])
    img.to_pil().save(f'results/{RESOLUTION_X}x{RESOLUTION_Y}-gradient.png')

if __name__ == "__main__":
    main()