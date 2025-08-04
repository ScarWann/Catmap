import array
import datashader as ds
from datashader import transfer_functions as tf
from enum import Enum
from numba import jit
import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg


MAX_ITERATIONS = 1000
MODES = Enum("MODE", "PERIODICITY_TO_START_GRADIENT PERIODICITY_TO_START PERIODICITY_PARTIAL_GRADIENT PERIODICITY_PARTIAL PIXEL_MOVEMENT")


@jit(nopython = True)
def periodicity_to_start_gradient(x_init: int, y_init: int, x_resolution: int, y_resolution: int) -> int:
    x = x_init
    y = y_init

    for i in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        if x == x_init and y == y_init:
            return i
    return -1

@jit(nopython = True)
def periodicity_to_start(x_init: int, y_init: int, x_resolution: int, y_resolution: int) -> int:
    x = x_init
    y = y_init

    for _ in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        if x == x_init and y == y_init:
            return 1
    return -1


@jit(nopython = True)
def periodicity_partial_gradient(x: int, y: int, x_resolution: int, y_resolution: int) -> int:
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
def periodicity_partial(x: int, y: int, x_resolution: int, y_resolution: int) -> int:
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


@jit(nopython = True)
def pixel_movement(x: int, y: int, x_resolution: int, y_resolution: int) -> np.ndarray:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype=np.int64)

    for _ in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        matrix[x, y] += 1
    return matrix



def apply_over_matrix(f, x_resolution: int, y_resolution: int) -> np.ndarray:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype=np.int16)

    for i in range(x_resolution):
        for j in range(y_resolution):
            matrix[i, j] = f(i, j, x_resolution, y_resolution)
    return matrix

def pack_rgba_data(matrix: np.ndarray):
    raveled_matrix = (matrix / matrix.max()).ravel()

    rgba_data = np.stack((
        raveled_matrix,
        np.zeros_like(raveled_matrix),
        np.zeros_like(raveled_matrix),
        np.ones_like(raveled_matrix)
    ), axis=1).reshape(-1)

    return array.array('f', rgba_data)

def pack_rgb_data(matrix: np.ndarray):
    raveled_matrix = (matrix / matrix.max()).ravel()

    rgba_data = np.stack((
        raveled_matrix,
        np.zeros_like(raveled_matrix),
        np.zeros_like(raveled_matrix)
    ), axis=1).reshape(-1)

    return array.array('f', rgba_data)

def visualize_matrix(matrix: list[list[int]], x_resolution: int, y_resolution: int, path: str) -> None:
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
    img.to_pil().save(path)

def log_visualizations(resolutions: list[list], mode = MODES.PERIODICITY_TO_START, start_coords = None) -> None:
    funcs = {
        MODES.PERIODICITY_TO_START_GRADIENT: periodicity_to_start_gradient,
        MODES.PERIODICITY_TO_START: periodicity_to_start,
        MODES.PERIODICITY_PARTIAL_GRADIENT: periodicity_partial_gradient,
        MODES.PERIODICITY_PARTIAL: periodicity_partial,
        MODES.PIXEL_MOVEMENT: pixel_movement
    }
    f = funcs[mode]

    for resolution in resolutions:
        if mode != MODES.PIXEL_MOVEMENT:
            matrix = apply_over_matrix(f, *resolution)
        else:
            matrix = f(*start_coords, *resolution)
        visualize_matrix(matrix, *resolution, path = f'results/{'x'.join(map(str, resolution))}-{mode.name}.png')


def main() -> None:
    WIDTH = 1920
    HEIGHT = 1080
    #log_visualizations(resolutions=[[200, 100]], mode = MODES.PIXEL_MOVEMENT, start_coords=(1, 2))
    dpg.create_context()
    matrix = apply_over_matrix(periodicity_to_start, HEIGHT, WIDTH)
    raw_data = pack_rgba_data(matrix)
    with dpg.texture_registry():
        dpg.add_raw_texture(
            width=WIDTH,
            height=HEIGHT,
            format=dpg.mvFormat_Float_rgba,
            default_value=raw_data,
            tag="matrix_texture"
        )

    with dpg.window(label="Main_image"):
        dpg.add_image("matrix_texture")

    with dpg.window(label="Control panel"):
        dpg.add_button(label="Save Image", callback=lambda:dpg.save_image(file=f"new{WIDTH}x{HEIGHT}.png", width=WIDTH, height=HEIGHT, data=pack_rgb_data(matrix), components=3))

    dpg.create_viewport(title='Catmap Visualizer', width=1920, height=1080)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()