import unittest
from pull_db_data import DBManager
from query_llm import LLMQueryer

test_message = [
    {
      "role": "user",
      "content": [{ "type": "text", "text": "knock knock." }]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "Who's there?" }]
    },
    {
      "role": "user",
      "content": [{ "type": "text", "text": "Orange." }]
    }
  ]

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

class TestLLMQueryer(unittest.TestCase):
    def setUp(self):
        self.llm_queryer = LLMQueryer(project_name='default')

    def test_set_project_name(self):
        self.llm_queryer.set_project_name("new_project_name")
        self.assertEqual(self.llm_queryer.project_name, "new_project_name")

    def test_query_chroma_db(self):
        query = "test query"
        n_results = 3
        results = self.llm_queryer.query_chroma_db(query, n_results)
        self.assertIsInstance(results, list)

    def test_query_postgres_db(self):
        connection_type = "local"
        query = "test query"
        n_results = 3
        results = self.llm_queryer.query_postgres_db(connection_type, query, n_results)
        self.assertIsInstance(results, list)

    def test_fetch_vectors_chroma(self):
        input_query = "test query"
        n_results = 3
        results = self.llm_queryer.fetch_vectors_chroma(input_query, n_results)
        self.assertIsInstance(results, str)

    def test_fetch_vectors_postgres(self):
        connection_type = "local"
        input_query = "test query"
        n_results = 3
        results = self.llm_queryer.fetch_vectors_postgres(connection_type, input_query, n_results)
        self.assertIsInstance(results, str)

    def test_query_llm(self):
        sys_msg = "test system message"
        include_rag = True
        results = self.llm_queryer.query_llm(sys_msg, test_message, include_rag)
        results = list(results)
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()
