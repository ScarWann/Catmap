CREATE_TABLE_CATMAP = """
    CREATE TABLE IF NOT EXISTS catmaps (
        id SERIAL PRIMARY KEY,
        width INTEGER,
        height INTEGER,
        xfunc varchar(255),
        yfunc varchar(255),
        xstart INTEGER,
        ystart INTEGER,
        type INTEGER
    );
"""

INSERT_CATMAP = """
    INSERT INTO catmaps (width, height, xfunc, yfunc, xstart, ystart, type)
    VALUES (%s, %s, %s, %s, %s, %s, %s);
"""

FIND_CATMAP = """
    SELECT id FROM catmaps 
    WHERE (width, height, xfunc, yfunc, xstart, ystart, type)
    = (%s, %s, %s, %s, %s, %s, %s);
"""

FIND_LAST_CATMAP_ID = """
    SELECT MAX(id) FROM catmaps;
"""

