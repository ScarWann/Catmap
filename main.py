import datashader as ds
from datashader import transfer_functions as tf
from enum import Enum
from numba import jit
import numpy as np
import pandas as pd


MAX_ITERATIONS = 1000
MODES = Enum("MODE", "PERIODICITY_TO_START_GRADIENT PERIODICITY_TO_START PERIODICITY_PARTIAL_GRADIENT PERIODICITY_PARTIAL PIXEL_MOVEMENT")


class CatmapFactory():
    def __init__(self, map: set = (lambda x, y: (2 * x + y),  lambda x, y: x + y)):
        self.map = map

    def return_catmap(self, resolution, start_coords = None) -> 'Catmap':
        return Catmap(self.map, resolution, start_coords)

class Catmap():
    def __init__(self: 'Catmap', map: set, resolution: set[int], start_coords: set[int] | None = None):
        self.x_func, self.y_func = map
        self.x_res, self.y_res = resolution
        if start_coords:
            self.start_x, self.start_y = start_coords
        else:
            self.start_x, self.start_y = None, None

    def __repr__(self):
        if self.start_x:
            return f'{self.x_res}x{self.y_res}-{self.start_x}-{self.start_y}'
        else:
            return f'{self.x_res}x{self.y_res}'
        
    def get_params(self):
        return self
        
    def get_id(self):
        df = pd.read_csv('logs.csv')
        return df[-1, 0]
        
        
    def save(self):
        id = pd.read_csv('logs.csv')[0]

        matrix = np.asarray(self.matrix)

        n, m = matrix.shape

        x_coords, y_coords = np.mgrid[0:n, 0:m]
        df = pd.DataFrame({
            'x': x_coords.ravel(),
            'y': y_coords.ravel(),
            'value': matrix.ravel()
        })

        canvas = ds.Canvas(plot_width=self.x_res, plot_height=self.y_res)
        agg = canvas.points(df, 'x', 'y', ds.mean('value'))
        img = tf.shade(agg, cmap=["black", "white"])
        img.to_pil().save(f"results/{id}.png")

    @jit(nopython = True)
    def periodicity_to_start_gradient(self) -> int:
        self.matrix = apply_over_matrix(periodicity_to_start_gradient, self.x_res, self.y_res)
    


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
        x, y = (x ** 2 + y) % x_resolution, (x + y) % y_resolution
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


@jit
def pixel_movement(x: int, y: int, x_resolution: int, y_resolution: int) -> np.ndarray:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype=np.int64)

    for _ in range(MAX_ITERATIONS):
        x, y = (2 * x + y) % x_resolution, (x + y) % y_resolution
        matrix[x, y] += 1
    return matrix




@jit(nopython = True)
def apply_over_matrix(f, x_resolution: int, y_resolution: int) -> np.ndarray:
    matrix = np.zeros(shape = (x_resolution, y_resolution), dtype = np.int16)
    
    for i in range(x_resolution):
        for j in range(y_resolution):
            matrix[i, j] = f(i, j, x_resolution, y_resolution)
    return matrix

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
    print(repr(lambda x: x * 2))
    #log_visualizations(resolutions=[[1920, 1080]], mode = MODES.PERIODICITY_TO_START, start_coords=(1, 2))
    #visualize_matrix(pixel_movement(2, 1, 200, 210), 200, 210, f'results/{'x'.join(map(str, [200, 210]))}-{'IDK'}.png')
    #log_visualizations([[24, 24]], mode = MODES.PERIODICITY_TO_START_GRADIENT)

if __name__ == "__main__":
    main()