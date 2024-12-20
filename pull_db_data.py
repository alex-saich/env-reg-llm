try:
    # Try to use pysqlite3 (for Streamlit Cloud)
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3 is not available, use built-in sqlite3 (local development)
    pass

import streamlit as st
import os
import psycopg2


def get_db_connection(connection_type='local'):
    if connection_type=='local':
        return psycopg2.connect(
            host=os.getenv('DB_HOST'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            sslmode='require'
        )
    elif connection_type=='streamlit':
            host=st.secrets["postgres"]["host"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            port=st.secrets["postgres"]["port"]

def pull_project_names():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT project_name FROM projects")
    project_names = [row[0] for row in cur.fetchall()]
    conn.close()
    return project_names

def insert_project_name(project_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO projects (project_name) VALUES (%s)", (project_name,))
        conn.commit()
        conn.close()
        return "Project name inserted successfully."
    except Exception as e:
        return f"Failed to insert project name: {e}"
    
def pull_project_pdfs(project_name):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT document_name FROM documents WHERE project_name = %s", (project_name,))
    pdfs = [row[0] for row in cur.fetchall()]
    conn.close()
    return pdfs


