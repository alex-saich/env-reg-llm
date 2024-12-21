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
import unittest

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
    
    def pull_project_pdfs(self, project_name):
        conn = self.get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT document_name FROM documents WHERE project_name = %s", (project_name,))
        pdfs = [row[0] for row in cur.fetchall()]
        conn.close()
        return pdfs

class TestDBManager(unittest.TestCase):
    def setUp(self):
        self.db_manager = DBManager()

    def test_get_db_connection_local(self):
        conn = self.db_manager.get_db_connection()
        self.assertIsNotNone(conn)

    def test_get_db_connection_streamlit(self):
        self.db_manager.connection_type = 'streamlit'
        conn = self.db_manager.get_db_connection()
        self.assertIsNotNone(conn)

    def test_pull_project_names(self):
        project_names = self.db_manager.pull_project_names()
        self.assertIsInstance(project_names, list)

    def test_insert_project_name(self):
        project_name = "Test Project"
        result = self.db_manager.insert_project_name(project_name)
        self.assertEqual(result, "Project name inserted successfully.")

    def test_pull_project_pdfs(self):
        project_name = "default"
        pdfs = self.db_manager.pull_project_pdfs(project_name)
        self.assertIsInstance(pdfs, list)

if __name__ == '__main__':
    unittest.main()


