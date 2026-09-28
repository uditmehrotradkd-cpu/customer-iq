from pathlib import Path
import sqlite3
from io import BytesIO
import pandas as pd

def load_csv(path=None, file_bytes=None):
    return pd.read_csv(BytesIO(file_bytes) if file_bytes is not None else path)

def load_excel(path=None, file_bytes=None, sheet_name=0):
    return pd.read_excel(BytesIO(file_bytes) if file_bytes is not None else path, sheet_name=sheet_name)

def sqlite_tables(path):
    con=sqlite3.connect(path)
    try:
        return pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con)['name'].tolist()
    finally: con.close()

def load_sqlite(path, table=None, query=None):
    con=sqlite3.connect(path)
    try:
        if query:
            return pd.read_sql_query(query, con)
        if not table:
            tables=sqlite_tables(path)
            if not tables: raise ValueError('SQLite database contains no tables.')
            table=tables[0]
        safe='"'+str(table).replace('"','""')+'"'
        return pd.read_sql_query(f'SELECT * FROM {safe}', con)
    finally: con.close()

def save_dataframe_to_sqlite(df, path, table='customers'):
    con=sqlite3.connect(path)
    try: df.to_sql(table, con, if_exists='replace', index=False)
    finally: con.close()
