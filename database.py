import psycopg2
from sql import *
from enum import Enum

class MODE(Enum):
    PERIODICITY_TO_START_GRADIENT = 1
    PERIODICITY_TO_START = 2
    PERIODICITY_PARTIAL_GRADIENT = 3
    PERIODICITY_PARTIAL = 4
    PIXEL_MOVEMENT = 5

try:
    conn = psycopg2.connect("dbname=catmaps user=dbuser host=localhost password=infected port=5432")
except:
    print("Unable to connect to the database")


def create_catmaps_table():
    with conn.cursor() as curs:
        curs.execute(CREATE_TABLE_CATMAP)
        conn.commit()

def insert_catmap_data(width: int, height: int, funcx: str, funcy: str, start_x: int, start_y: int, maptype: MODE):
    with conn.cursor() as curs:
        curs.execute(INSERT_CATMAP, (width, height, funcx, funcy, start_x, start_y, maptype.value))
        conn.commit()

def find_catmap_id(width: int, height: int, funcx: str, funcy: str, start_x: int, start_y: int, maptype: MODE):
    with conn.cursor() as curs:
        curs.execute(FIND_CATMAP, (width, height, funcx, funcy, start_x, start_y, maptype.value))
        id_ = curs.fetchone()
        return id_

def find_last_id():
    with conn.cursor() as curs:
        curs.execute(FIND_LAST_CATMAP_ID)
        id_ = curs.fetchone()
        return id_[0] + 1 if id_[0] else 1