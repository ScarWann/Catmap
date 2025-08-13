#!/usr/bin/env python3

import array
import datashader as ds
from datashader import transfer_functions as tf
from numba import njit, core
import numpy as np
import pandas as pd
from PIL import Image
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


@njit
def apply_over_matrix(f: core.registry.CPUDispatcher, x_resolution: int, y_resolution: int, x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher) -> np.ndarray:
    matrix = np.zeros(shape = (y_resolution, x_resolution), dtype=np.int16)

    for i in range(x_resolution):
        for j in range(y_resolution):
            matrix[j, i] = f(i, j, x_resolution, y_resolution, x_func, y_func)
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
                   x_start: int = 0, y_start: int = 0, 
                   x_func_str: str = "2 * x + y", y_func_str: str = "x + y") -> np.ndarray:
    
    try:
        id = find_catmap_id(x_resolution, y_resolution, x_func_str, y_func_str, x_start, y_start, mode)[0]
        im_frame = Image.open(f"{id}_{x_resolution}x{y_resolution}.png")
        np_frame = np.array(im_frame.getdata(0))
        return np_frame / np_frame.max()
    except:
        id = find_last_id()


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

        return matrix


def update_image(sender, app_data, user_data):
    print(user_data)
    if user_data:
        width, height, x_function, y_function, x_start, y_start = user_data[:-1]
        mode = MODE(user_data[-1])
    else:
        width = dpg.get_value("generation width resolution")
        height = dpg.get_value("generation height resolution")
        x_start = dpg.get_value("generation x start")
        y_start = dpg.get_value("generation y start")
        x_function = dpg.get_value("generation x function")
        y_function = dpg.get_value("generation y function")
        mode = dpg.get_value("generation mode")
        gradient = dpg.get_value("generation gradient")

        modes = {
            (False, "Periodicity to start"): MODE.PERIODICITY_TO_START,
            (True, "Periodicity to start"): MODE.PERIODICITY_TO_START_GRADIENT,
            (False, "General periodicity"): MODE.PERIODICITY_PARTIAL,
            (True, "General periodicity"): MODE.PERIODICITY_PARTIAL_GRADIENT,
            (False, "Individual pixel movement"): MODE.PIXEL_MOVEMENT,
            (True, "Individual pixel movement"): MODE.PIXEL_MOVEMENT
        }

        mode = modes[(gradient, mode)]



    matrix = generate_image(width, height, mode, x_start, y_start, x_function, y_function)
    raw_data = pack_rgb_data(matrix)

    dpg.set_value("matrix_texture", raw_data)

def test(sender, app_data, user_data):
    print(sender, app_data, user_data)

def update_generating_panel(sender, app_data):
    if app_data == "Individual pixel movement":
        dpg.show_item("generation pixel movement")
        dpg.hide_item("generation gradient")
    else:
        dpg.hide_item("generation pixel movement")
        dpg.show_item("generation gradient")

def create_generating_panel():
    with dpg.window(label="Generation panel"):
        with dpg.group(horizontal=True):
            dpg.add_input_int(label = "Image width", default_value=1920, tag = "generation width resolution", width=100)
            dpg.add_input_int(label = "Image height", default_value=1080, tag = "generation height resolution", width=100)
        with dpg.group(horizontal=True):
            dpg.add_input_text(label = "X function", default_value="2 * x + y", tag = "generation x function", width=100)
            dpg.add_input_text(label = "Y function", default_value="x + y", tag = "generation y function", width=100)
        dpg.add_combo(items=["Periodicity to start", "General periodicity", "Individual pixel movement"], default_value="Periodicity to start", tag = "generation mode", callback=update_generating_panel)
        with dpg.group(horizontal=True, tag = "generation pixel movement", show = False):
            dpg.add_input_int(label = "Starting X", default_value=0, tag = "generation x start", width=100)
            dpg.add_input_int(label = "Starting Y", default_value=0, tag = "generation y start", width=100)
        dpg.add_checkbox(label="Gradient mode (displays how fast a pixel returns to original position, for example)", default_value=False, tag = "generation gradient")
        dpg.add_button(label="Generate", callback=update_image)

def create_database_panel():
    with dpg.window(label = "Saved catmaps"):
        with dpg.table(header_row=True):

            dpg.add_table_column(label="ID")
            dpg.add_table_column(label="Width")
            dpg.add_table_column(label="Height")
            dpg.add_table_column(label="X function")
            dpg.add_table_column(label="Y function")
            dpg.add_table_column(label="X start")
            dpg.add_table_column(label="Y start")
            dpg.add_table_column(label="Mode")
            dpg.add_table_column()

            data = load_catmaps_table()
            print(data)
            for catmap in data:
                with dpg.table_row():
                    for param in catmap:
                        dpg.add_text(param)
                    dpg.add_button(label = "Load", callback = update_image, user_data=catmap[1:])



def create_pixel_panel():
    pass

def create_help_panel():
    pass

def create_legend_panel():
    pass

def setup():
    with dpg.handler_registry():
        dpg.add_key_press_handler(key=dpg.mvKey_C, callback=create_generating_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_D, callback=create_database_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_X, callback=create_pixel_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_H, callback=create_help_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_C, callback=create_legend_panel)

    


def main() -> None:
    WIDTH = 1920
    HEIGHT = 1080

    dpg.create_context()
    setup()
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

    """with dpg.window(no_move=True, no_resize=True, no_inputs=True, no_title_bar=True, no_background=True):
        with dpg.drawlist(width=1920, height=1080):
            dpg.draw_image("matrix_texture", (0, 0), (1920, 1080))"""

    dpg.create_viewport(title='Catmap Visualizer', width=1920, height=1080, decorated=False)

    with dpg.window(label="Main_image", no_move=True, no_title_bar=True, no_bring_to_front_on_focus=True, no_scrollbar=True, no_scroll_with_mouse=True, no_resize=True, no_background=True, no_collapse=True) as background_window:
        dpg.add_image("matrix_texture")
        with dpg.theme() as borderless_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
            dpg.bind_item_theme(background_window, borderless_theme)

    def resize_to_viewport():
        w = dpg.get_viewport_client_width()
        h = dpg.get_viewport_client_height()
        dpg.set_item_width(background_window, w)
        dpg.set_item_height(background_window, h)

    dpg.set_viewport_resize_callback(lambda s,a: resize_to_viewport())

    #dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    resize_to_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()