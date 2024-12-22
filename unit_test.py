import unittest
from pull_db_data import DBManager

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

    def test_delete_project_name(self):
        project_name = "Test Project"
        # First, insert the project name to ensure it exists
        self.db_manager.insert_project_name(project_name)
        # Now, delete the project name
        result = self.db_manager.delete_project_name(project_name)
        self.assertEqual(result, "Project name deleted successfully.")
        # Verify that the project name has been deleted
        project_names = self.db_manager.pull_project_names()
        self.assertNotIn(project_name, project_names)

if __name__ == '__main__':
    unittest.main()
