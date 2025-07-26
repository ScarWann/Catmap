import datashader as ds
from datashader import transfer_functions as tf
from enum import Enum
from numba import jit
import numpy as np
import pandas as pd
from time import time


MAX_ITERATIONS = 1000
MODES = Enum("MODE", "PERIODICITY_TO_START_GRADIENT PERIODICITY_TO_START PERIODICITY_PARTIAL_GRADIENT PERIODICITY_PARTIAL PIXEL_MOVEMENT")


@jit(nopython = True)
def periodicity_to_start_gradient(x_init, y_init, x_resolution, y_resolution):
    x = x_init
    y = y_init
    for i in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        if x == x_init and y == y_init:
            return i
    return -1

@jit(nopython = True)
def periodicity_to_start(x_init, y_init, x_resolution, y_resolution):
    x = x_init
    y = y_init
    for i in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        if x == x_init and y == y_init:
            return 1
    return -1

@jit(nopython = True)
def periodicity_partial_gradient_new(x, y, x_resolution, y_resolution):
    visited = {}
    state = (x, y)
    visited[state] = 0

    for i in range(1, MAX_ITERATIONS + 1):
        x_next = (2 * x + y) % x_resolution
        y_next = (x + y) % y_resolution
        state = (x, y) = (x_next, y_next)

        if state in visited:
            return i - visited[state]
        visited[state] = i
    return -1

@jit(nopython = True)
def periodicity_partial_new(x, y, x_resolution, y_resolution):
    visited = {}
    state = (x, y)
    visited[state] = 0

    for _ in range(1, MAX_ITERATIONS + 1):
        x_next = (2 * x + y) % x_resolution
        y_next = (x + y) % y_resolution
        state = (x, y) = (x_next, y_next)

        if state in visited:
            return 1
        visited[state] = 0
    return -1

@jit
def periodicity_partial_gradient_old(x, y, x_resolution, y_resolution):
    visited = [[x, y]]
    for i in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        if [x, y] in visited:
            return i - visited.index([x, y])
        visited.append([x,y])
    return -1

@jit
def periodicity_partial_old(x, y, x_resolution, y_resolution):
    visited = [[x, y]]
    for _ in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        if [x, y] in visited:
            return 1
        visited.append([x,y])
    return -1

@jit(nopython = True)
def pixel_movement(x, y, x_resolution, y_resolution):
    matrix = np.zeros((x_resolution, y_resolution), int)
    for _ in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        matrix[x][y] += 1
    return matrix


@jit(nopython = True)
def apply_over_matrix(f, x_resolution, y_resolution):
    matrix = []
    for i in range(x_resolution):
        row = []
        for j in range(y_resolution):
            row.append(f(i, j, x_resolution, y_resolution))
        matrix.append(row)
    return matrix

def visualize_matrix(matrix, x_resolution, y_resolution, path):
    matrix = np.asarray(matrix)

    n, m = matrix.shape

    x_coords, y_coords = np.mgrid[0:n, 0:m]
    df = pd.DataFrame({
        'x': x_coords.ravel(),
        'y': y_coords.ravel(),
        'value': matrix.ravel()
    })

    canvas = ds.Canvas(plot_width=x_resolution, plot_height=y_resolution)
    agg = canvas.points(df, 'x', 'y', ds.mean('value'))
    img = tf.shade(agg, cmap=["black", "white"])
    img.to_pil().save()

def log_visualizations(resolutions: list[list], mode = MODES.PERIODICITY_TO_START):
    funcs = {
        MODES.PERIODICITY_TO_START_GRADIENT: periodicity_to_start_gradient,
        MODES.PERIODICITY_TO_START: periodicity_to_start,
        MODES.PERIODICITY_PARTIAL_GRADIENT: periodicity_partial_gradient_new,
        MODES.PERIODICITY_PARTIAL: periodicity_partial_new
    }
    f = funcs[mode]
    for resolution in resolutions:
        matrix = apply_over_matrix(f, *resolution)
        visualize_matrix(matrix, *resolution, path = f'results/{str(resolution)}-{mode.name}.png')


def main():
    #log_visualizations(resolutions=[[200, 210]], mode = MODES.PERIODICITY_PARTIAL)
    start_time = time()
    a = apply_over_matrix(periodicity_partial_new, 1000, 500)
    print(time() - start_time)
    start_time = time()
    a = apply_over_matrix(periodicity_partial_old, 1000, 500)
    print(time() - start_time)
    start_time = time()
    a = apply_over_matrix(periodicity_partial_gradient_new, 1000, 500)
    print(time() - start_time)
    start_time = time()
    a = apply_over_matrix(periodicity_partial_gradient_old, 1000, 500)
    print(time() - start_time)

if __name__ == "__main__":
    main()