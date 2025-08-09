import psycopg2

try:
    conn = psycopg2.connect("dbname='catmaps' user='dbuser' host='localhost' password='infected' port='5432'")
except:
    print("I am unable to connect to the database")


with conn.cursor() as curs:
    pass