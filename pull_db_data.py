try:
    # Try to use pysqlite3 (for Streamlit Cloud)
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3 is not available, use built-in sqlite3 (local development)
    pass

import streamlit as st
from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

class DBManager:
    def __init__(self, connection_type='local'):
        self.connection_type = connection_type

    def get_db_connection(self):
        try:
            if self.connection_type == 'local':
                return psycopg2.connect(
                    host=os.getenv('DB_HOST'),
                    dbname=os.getenv('DB_NAME'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD'),
                    sslmode='require'
                )
            elif self.connection_type == 'streamlit':
                return psycopg2.connect(
                    host=st.secrets["postgres"]["host"],
                    dbname=st.secrets["postgres"]["database"],
                    user=st.secrets["postgres"]["user"],
                    password=st.secrets["postgres"]["password"],
                    port=st.secrets["postgres"]["port"]
                )
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None  # Explicitly return None on failure

    def pull_project_names(self):
        conn = self.get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT project_name FROM projects")
        project_names = [row[0] for row in cur.fetchall()]
        conn.close()
        return project_names

    def insert_project_name(self, project_name):
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            cur.execute("INSERT INTO projects (project_name) VALUES (%s)", (project_name,))
            conn.commit()
            conn.close()
            return "Project name inserted successfully."
        except Exception as e:
            return f"Failed to insert project name: {e}"

    def delete_project_name(self, project_name):
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM projects WHERE project_name = %s", (project_name,))
            conn.commit()
            conn.close()
            return "Project name deleted successfully."
        except Exception as e:
            return f"Failed to delete project name: {e}"
    
    def pull_project_pdfs(self, project_name):
        conn = self.get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT document_name, document_summary FROM documents WHERE project_name = %s", (project_name,))
        pdfs = [(row[0], row[1]) for row in cur.fetchall()]
        conn.close()
        return pdfs

def insert_project_name(self, project_name):
    try:
        conn = self.get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO projects (project_name) VALUES (%s)", (project_name,))
        conn.commit()
        conn.close()
        return "Project name inserted successfully."
    except Exception as e:
        return f"Failed to insert project name: {e}"
    
def pull_project_pdfs(self, project_name):
    conn = self.get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT document_name FROM documents WHERE project_name = %s", (project_name,))
    pdfs = [row[0] for row in cur.fetchall()]
    conn.close()
    return pdfs


