from store_pdfs_pg import get_db_connection

def pull_project_names():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT project_name FROM projects")
    project_names = [row[0] for row in cur.fetchall()]
    conn.close()
    return project_names

