## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Explores (and extracts in later step) the soil information from the BUEK200 database. Uses functions of the MONICA
## repository from ZALF e.V. (the soil_io3 script)

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import sqlite3
from sqlite3 import Error
import pandas as pd
import geopandas as gpd
import os

import soil_io3

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


def excel_files_to_sqlit_db():
    # create_connection(r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\buek200.db")

    ## Create new sqlite database
    conn = sqlite3.connect(r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\buek200_data.sqlite")
    c = conn.cursor()

    ## Load excel tables and put them into sqlite db
    pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\BUEK200_Sachdaten\tblBlattlegendeneinheit.xlsx"
    df = pd.read_excel(pth)
    df.to_sql('tblBlattlegendeneinheit', conn, if_exists='append', index=False)

    pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\BUEK200_Sachdaten\tblHorizonte.xlsx"
    df = pd.read_excel(pth)
    df.to_sql('tblHorizonte', conn, if_exists='append', index=False)

    pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\BUEK200_Sachdaten\tblProfile.xlsx"
    df = pd.read_excel(pth)
    df.to_sql('tblProfile', conn, if_exists='append', index=False)

    pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\BUEK200_Sachdaten\tblZuordnungGLE_BLE.xlsx"
    df = pd.read_excel(pth)
    df.to_sql('tblZuordnungGLE_BLE', conn, if_exists='append', index=False)

    conn.close()
# soil_db_con = sqlite3.connect(paths["path-to-data-dir"] + DATA_SOIL_DB)

def soil_vector_ids_to_soil_profiles():
    pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\buek200_buffered_3m.shp"
    gdf = gpd.read_file(pth)

    pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\buek200\buek200_old.sqlite"
    con = sqlite3.connect(pth)

    # t = pd.read_sql_query(f"SELECT * from soil_profile", con)
    # t2 = pd.read_sql_query(f"SELECT * from soil_profile_all", con)
    #
    # ## Getting all tables from sqlite_master
    # sql_query = """SELECT name FROM sqlite_master WHERE type='ta';"""
    # cursor = con.cursor()
    # cursor.execute(sql_query)
    # table_name_lst = cursor.fetchall()
    # print("List of tables\n", table_name_lst)
    #
    # for table_name in table_name_lst:
    #     print(table_name[0])
    #     table = pd.read_sql_query(f"SELECT * from {table_name[0]}", con)
    #     print(table)

    soil_id = 151801
    soil_profile = soil_io3.soil_parameters(con, soil_id)

    for i in range(10):
        soil_id = gdf['TKLE_NR'].iloc[i]
        # soil_id = int(soil_grid[srow, scol])
        soil_profile = soil_io3.soil_parameters(con, str(soil_id))
        print(i, f"soil id: {soil_id}", f"number profiles: {len(soil_profile)}\n", soil_profile)




def main():
    # excel_files_to_sqlit_db()
    soil_vector_ids_to_soil_profiles()

if __name__ == '__main__':
    main()

