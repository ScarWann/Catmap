import array
import datashader as ds
from datashader import transfer_functions as tf
from numba import njit, core
import numpy as np
import pandas as pd
import database
from database import *
import dearpygui.dearpygui as dpg
from time import perf_counter


MAX_ITERATIONS = 1000

def make_njit_func(func_str: str):
    src = f"def f(x, y):\n    return {func_str}"
    ns = {}
    exec(src, ns)
    return njit(ns["f"])

@njit
def periodicity_to_start_gradient(x_init: int, y_init: int, x_resolution: int, y_resolution: int, x_func, y_func) -> int:
    x = x_init
    y = y_init

    for i in range(MAX_ITERATIONS):
        x, y = x_func(x, y) % x_resolution, y_func(x, y) % y_resolution
        if x == x_init and y == y_init:
            return i
    return -1

@njit
def periodicity_to_start(x_init: int, y_init: int, x_resolution: int, y_resolution: int, x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher) -> int:
    x = x_init
    y = y_init

    for _ in range(MAX_ITERATIONS):
        x, y = x_func(x, y) % x_resolution, y_func(x, y) % y_resolution
        if x == x_init and y == y_init:
            return 1
    return -1

@njit
def periodicity_to_start_exec(x_init: int, y_init: int, x_resolution: int, y_resolution: int, x_func: str = "2 * x + y", y_func: str = "x + y") -> int:
    x = x_init
    y = y_init

    for _ in range(MAX_ITERATIONS):
        x, y = x_func(x, y) % x_resolution, y_func(x, y) % y_resolution
        if x == x_init and y == y_init:
            return 1
    return -1


@njit
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

@njit
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


@njit
def pixel_movement(x: int, y: int, x_resolution: int, y_resolution: int) -> np.ndarray:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype=np.int64)

    for _ in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        matrix[x, y] += 1
    return matrix



def apply_over_matrix(f: core.registry.CPUDispatcher, x_resolution: int, y_resolution: int, x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher) -> np.ndarray:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype=np.int16)

    for i in range(x_resolution):
        for j in range(y_resolution):
            matrix[i, j] = f(i, j, x_resolution, y_resolution, x_func, y_func)
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
    raveled_matrix = (matrix / matrix.max() * 255).ravel()

    rgba_data = np.stack((
        raveled_matrix,
        np.zeros_like(raveled_matrix),
        np.zeros_like(raveled_matrix)
    ), axis=1).reshape(-1)

    rgba_data[rgba_data < 0] = 0

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

def log_visualizations(resolutions: list[list], mode = MODE.PERIODICITY_TO_START, start_coords = None) -> None:
    funcs = {
        MODE.PERIODICITY_TO_START_GRADIENT: periodicity_to_start_gradient,
        MODE.PERIODICITY_TO_START: periodicity_to_start,
        MODE.PERIODICITY_PARTIAL_GRADIENT: periodicity_partial_gradient,
        MODE.PERIODICITY_PARTIAL: periodicity_partial,
        MODE.PIXEL_MOVEMENT: pixel_movement
    }
    f = funcs[mode]

    for resolution in resolutions:
        if mode != MODE.PIXEL_MOVEMENT:
            matrix = apply_over_matrix(f, *resolution)
        else:
            matrix = f(*start_coords, *resolution)
        visualize_matrix(matrix, *resolution, path = f'results/{'x'.join(map(str, resolution))}-{mode.name}.png')

def generate_image(x_resolution: int = 1920, y_resolution: int = 1080,
                   mode: MODE = MODE.PERIODICITY_TO_START,
                   x_start: int | None = None, y_start: int | None = None, 
                   x_func_str: str = "2 * x + y", y_func_str: str = "x + y") -> np.ndarray:
    
    create_catmaps_table()
    id = find_catmap_id(x_resolution, y_resolution, x_func_str, y_func_str, x_start, y_start, mode)
    if not id:
        id = find_last_id()

    print("PostgreSQL id recieved")

    x_func = make_njit_func(x_func_str)
    y_func = make_njit_func(y_func_str)

    print("Njitification successful")

    funcs = {
        MODE.PERIODICITY_TO_START_GRADIENT: periodicity_to_start_gradient,
        MODE.PERIODICITY_TO_START: periodicity_to_start,
        MODE.PERIODICITY_PARTIAL_GRADIENT: periodicity_partial_gradient,
        MODE.PERIODICITY_PARTIAL: periodicity_partial,
        MODE.PIXEL_MOVEMENT: pixel_movement
    }
    f = funcs[mode]


    if mode != MODE.PIXEL_MOVEMENT:
        matrix = apply_over_matrix(f, x_resolution, y_resolution, x_func, y_func)
    else:
        matrix = f(x_start, y_start, x_resolution, y_resolution)

    print("Matrix generation successful")

    dpg.save_image(file=f"{id}_{x_resolution}x{y_resolution}.png", width=x_resolution, height=y_resolution, data=pack_rgb_data(matrix), components=3)

    print("Save complete")

    insert_catmap_data(x_resolution, y_resolution, x_func_str, y_func_str, x_start, y_start, mode)

    print("DB logging complete")



def main() -> None:

    WIDTH = 1920
    HEIGHT = 1080
    #log_visualizations(resolutions=[[200, 100]], mode = MODES.PIXEL_MOVEMENT, start_coords=(1, 2))
    dpg.create_context()
    matrix = generate_image()
    raw_data = pack_rgb_data(matrix)
    with dpg.texture_registry():
        dpg.add_raw_texture(
            width=WIDTH,
            height=HEIGHT,
            format=dpg.mvFormat_Float_rgb,
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

    """
    start = perf_counter()
    matrix = apply_over_matrix(periodicity_to_start, HEIGHT, WIDTH)
    end = perf_counter()
    print(end - start)
    start = perf_counter()
    matrix = apply_over_matrix(periodicity_to_start_exec, HEIGHT, WIDTH)
    end = perf_counter()
    print(end - start)"""


if __name__ == "__main__":
    main()