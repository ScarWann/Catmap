#!/usr/bin/env python3

import array
import database
from database import *
import dearpygui.dearpygui as dpg
from numba import njit, core
import numpy as np
from PIL import Image
from time import perf_counter
import math


PRECISION = 1
calculate_pixel_movement_on_press = True

def make_njit_func(func_str: str):
    src = f"def f(x, y):\n    return {func_str}"
    ns = {}
    exec(src, ns)
    return njit(ns["f"])

@njit
def periodicity_to_start_gradient(x_init: int, y_init: int, 
                                  x_resolution: int, y_resolution: int, 
                                  x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher,
                                  iterations: int) -> int:
    x = x_init
    y = y_init

    for i in range(iterations):
        x, y = x_func(x, y) % x_resolution, y_func(x, y) % y_resolution
        if x == x_init and y == y_init:
            return i
    return -1

@njit
def periodicity_to_start(x_init: int, y_init: int, 
                         x_resolution: int, y_resolution: int, 
                         x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher,
                         iterations: int) -> int:
    x = x_init
    y = y_init

    for _ in range(iterations):
        x, y = x_func(x, y) % x_resolution, y_func(x, y) % y_resolution
        if x == x_init and y == y_init:
            return 1
    return -1

@njit
def periodicity_partial_gradient(x: int, y: int, x_resolution: int, y_resolution: int, 
                                 x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher,
                                 iterations: int) -> int:
    visited = {}
    state = (x, y)
    visited[state] = 0

    for i in range(1, iterations + 1):
        x_next = x_func(x, y) % x_resolution
        y_next = y_func(x, y) % y_resolution
        state = (x, y) = (x_next, y_next)

        if state in visited:
            return i - visited[state]
        visited[state] = i
    return -1

@njit
def periodicity_partial(x: int, y: int, x_resolution: int, y_resolution: int, 
                        x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher,
                        iterations: int) -> int:
    visited = {}
    state = (x, y)
    visited[state] = 1

    for i in range(1, iterations + 1):
        x_next = x_func(x, y) % x_resolution
        y_next = y_func(x, y) % y_resolution
        state = (x, y) = (x_next, y_next)

        if state in visited:
            return i - visited[state]
        visited[state] = 1
    return -1


@njit
def pixel_movement_brute(x: int, y: int, x_resolution: int, y_resolution: int, 
                         x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher,
                         iterations: int) -> int:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype=np.int64)

    for _ in range(iterations):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        matrix[x, y] += 1
    return matrix

@njit
def pixel_movement(x: int, y: int, x_resolution: int, y_resolution: int, 
                   x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher,
                   iterations: int) -> int:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype=np.int64)

    for _ in range(iterations):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        matrix[x, y] += 1
    return matrix

@njit
def apply_over_matrix(f: core.registry.CPUDispatcher, x_resolution: int, y_resolution: int, 
                      x_func: core.registry.CPUDispatcher, y_func: core.registry.CPUDispatcher, 
                      segmented: bool = False,
                      x_min: int = 0, y_min: int = 0, 
                      x_max: int = 0, y_max: int = 0) -> np.ndarray:
    matrix = np.zeros(shape = (y_resolution, x_resolution), dtype=np.int16)
    iterations = (min(x_max, x_resolution) - max(0, x_min)) * (min(y_max, y_resolution) - max(0, y_min)) * PRECISION
    if segmented:
        for i in range(max(0, x_min), min(x_max, x_resolution)):
            for j in range(max(0, y_min), min(y_max, y_resolution)):
                matrix[j, i] = f(i, j, x_resolution, y_resolution, x_func, y_func, iterations)
        return matrix
    else:
        for i in range(x_resolution):
            for j in range(y_resolution):
                matrix[j, i] = f(i, j, x_resolution, y_resolution, x_func, y_func, iterations)
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

@njit
def pack_rgb_data(matrix: np.ndarray):
    raveled_matrix = (matrix / matrix.max() * 255).ravel()

    rgba_data = np.stack((
        raveled_matrix,
        np.zeros_like(raveled_matrix),
        np.zeros_like(raveled_matrix)
    ), axis=1).reshape(-1)

    rgba_data[rgba_data < 0] = 0

    #return array.array('f', rgba_data)
    return rgba_data

@njit
def fit_to_screen_size(matrix: np.ndarray):
    SCREEN_HEIGHT = 1080
    SCREEN_WIDTH = 1920

    matrix_height, matrix_width = matrix.shape
    while True:
        if matrix_height * 2 <= SCREEN_HEIGHT and matrix_width * 2 <= SCREEN_WIDTH:
            temp_matrix = np.repeat(matrix, repeats=2, axis=1)
            matrix = np.repeat(temp_matrix, repeats=2, axis=0)
            matrix_height *= 2
            matrix_width *= 2
        elif matrix_height > SCREEN_HEIGHT or matrix_width > SCREEN_WIDTH:
            sh = (matrix.shape[0] // 2, 2, matrix.shape[1] // 2, 2)
            matrix = matrix.reshape(sh).mean(axis = (1, 3))
            matrix_height = math.floor(float(matrix_height) / 2)
            matrix_width = math.floor(float(matrix_width) / 2)
        else:
            break
    padded = np.pad(matrix, pad_width=((math.floor((SCREEN_HEIGHT - matrix_height) / 2),\
                                        math.ceil((SCREEN_HEIGHT - matrix_height) / 2)),\
                                       (math.floor((SCREEN_WIDTH - matrix_width) / 2),  \
                                        math.ceil((SCREEN_WIDTH - matrix_width) / 2))), \
                                        mode='constant', constant_values=0)
    
    return padded
    
def fetch_image(x_resolution, y_resolution, x_func_str, y_func_str, x_start, y_start, mode):
    time = perf_counter()
    id_ = find_catmap_id(x_resolution, y_resolution, x_func_str, y_func_str, x_start, y_start, mode)[0]
    print(f"Step speed {perf_counter() - time}")
    time = perf_counter()
    im_frame = Image.open(f"{id_}_{x_resolution}x{y_resolution}.png")
    print(f"Step speed {perf_counter() - time}")
    time = perf_counter()
    np_frame = np.array(im_frame.getdata(0), dtype=np.float16)
    print(f"Step speed {perf_counter() - time}")
    time = perf_counter()
    np_frame = np.reshape(np_frame, (y_resolution, x_resolution))
    print(f"Step speed {perf_counter() - time}")
    time = perf_counter()
    return id_, np_frame / np_frame.max()

def generate_image(x_resolution: int = 1920, y_resolution: int = 1080,
                   mode: MODE = MODE.PERIODICITY_TO_START,
                   x_start: int = 0, y_start: int = 0, 
                   x_func_str: str = "2 * x + y", y_func_str: str = "x + y"):
    
    try:
        time = perf_counter()
        data = fetch_image(x_resolution, y_resolution, x_func_str, y_func_str, x_start, y_start, mode)
        print(f"Fetch speed {perf_counter() - time}")
        return data
    except:
        id_ = find_last_id()

        time = perf_counter()
        x_func = make_njit_func(x_func_str)
        y_func = make_njit_func(y_func_str)

        print(f"Njitification successful, speed:{perf_counter() - time}")

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

        print(f"Matrix generation successful, speed:{perf_counter() - time}")

        dpg.save_image(file=f"{id_}_{x_resolution}x{y_resolution}.png", width=x_resolution, height=y_resolution, data=pack_rgb_data(matrix), components=3)

        time = perf_counter()
        print(f"Save complete, speed:{perf_counter() - time}")

        insert_catmap_data(x_resolution, y_resolution, x_func_str, y_func_str, x_start, y_start, mode)

        print("DB logging complete")

        return id_, matrix


def update_image(sender, app_data, user_data):
    time = perf_counter()
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



    print(f"Setup speed: {perf_counter() - time}")
    id_, matrix = generate_image(width, height, mode, x_start, y_start, x_function, y_function)
    time = perf_counter()
    raw_data = pack_rgb_data(fit_to_screen_size(matrix))
    print(f"Packing speed: {perf_counter() - time}")

    
    time = perf_counter()
    dpg.set_value("matrix_texture", raw_data)
    print(f"Rendering speed: {perf_counter() - time}")

    return id_

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

def create_database_panel(sender, app_data, user_data):
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
            for catmap in data:
                with dpg.table_row():
                    for param in catmap:
                        dpg.add_text(param)
                    dpg.add_button(label = "Load", callback = update_image, user_data=catmap[1:])

def switch_pixel_mode():
    global calculate_pixel_movement_on_press
    calculate_pixel_movement_on_press = not calculate_pixel_movement_on_press

def create_pixel_panel():
    
    while dpg.is_dearpygui_running() and not dpg.is_key_pressed(key=dpg.mvKey_S):
        x_pos, y_pos = dpg.get_mouse_pos(local = False)
        print(f"X pos: {x_pos}, Y pos: {y_pos}")

def create_help_panel():
    pass

def create_legend_panel():
    pass

def create_zoom_panel():
    pass

def setup():
    with dpg.handler_registry():
        dpg.add_key_press_handler(key=dpg.mvKey_C, callback=create_generating_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_D, callback=create_database_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_X, callback=create_pixel_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_H, callback=create_help_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_C, callback=create_legend_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_Z, callback=create_zoom_panel)
        dpg.add_key_press_handler(key=dpg.mvKey_M, callback=switch_pixel_mode)

    


def main() -> None:
    WIDTH = 1920
    HEIGHT = 1080

    dpg.create_context()
    setup()
    raw_data = array.array('f', np.zeros((WIDTH * HEIGHT * 3), dtype=np.float16))
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

    with dpg.window(label="Main_image", no_move=True, no_title_bar=True, no_bring_to_front_on_focus=True, no_scroll_with_mouse=True, no_scrollbar=True, no_resize=True, no_background=True, no_collapse=True) as background_window:
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