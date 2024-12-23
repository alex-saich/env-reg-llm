try:
    # Try to use pysqlite3 (for Streamlit Cloud)
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3 is not available, use built-in sqlite3 (local development)
    pass

from dotenv import load_dotenv
import langchain_openai 
import langchain_core
import langchain_chroma 
import chromadb
import os
import streamlit as st
from pull_db_data import DBManager

class LLMQueryer:
    def __init__(self, project_name, connection_type='local', embedding_model=None):
        self.is_debug = os.getenv('DEBUG')
        self.project_name = project_name
        self.connection_type = connection_type

        self.openai_api_key = None
        
        # Set up OpenAI API key based on environment
        if os.environ["OPENAI_API_KEY"] is None:
            if connection_type == 'local':
                load_dotenv()
                self.openai_api_key = os.getenv('OPENAI_API_KEY')
            else:
                try:
                    self.openai_api_key = st.secrets['openai']['api_key']
                    print(f"Retrieved API key from secrets: {self.openai_api_key[:8]}...")  # Only print first 8 chars for security
                except Exception as e:
                    print(f"Error accessing secrets: {str(e)}")
                    raise
            
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not found in environment variables or secrets")
            
            # Initialize OpenAI client with appropriate API key
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

        # Initialize embedding model
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            try:
                self.embedding_model = langchain_openai.OpenAIEmbeddings(
                    # openai_api_key=self.openai_api_key,
                    model="text-embedding-ada-002"
                )
            except Exception as e:
                print(f"Error initializing embedding model: {str(e)}")
                raise


    def set_project_name(self, new_project_name):
        self.project_name = new_project_name
    
    def query_chroma_db(self, query, n_results=3, client=None, chroma_db=None):
        if client is None:
            client = chromadb.PersistentClient(path="vector_db")
        if chroma_db is None:
            chroma_db = langchain_chroma.Chroma(collection_name=self.project_name, embedding_function=self.embedding_model, persist_directory="vector_db")

        results = chroma_db.similarity_search(query, n_results)
        
        if self.is_debug:
            print(str(type(results)))
            print(str(len(results)))

        return results

    def query_postgres_db(self, connection_type, query, n_results=3):
        try:
            conn = DBManager(connection_type).get_db_connection()
            cur = conn.cursor()
        except Exception as e:
            if self.is_debug:
                print(f"Error connecting to database: {str(e)}")
            raise e

        # Generate an embedding for the query input
        query_embedding = self.embedding_model.embed_query(query)

        cur.execute(
            """
            SELECT chunk_text, metadata
                FROM doc_vectors
                WHERE project_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
            (self.project_name, query_embedding, n_results)
        )
        results = cur.fetchall()

        if self.is_debug:
            print(str(type(results)))
            print(str(len(results)))

        cur.close()
        conn.close()

        return results

    def fetch_vectors_chroma(self, input_query, n_results=3):

        client = chromadb.PersistentClient(path="vector_db")
        chroma_db = langchain_chroma.Chroma(client=client, collection_name=self.project_name, embedding_function=self.embedding_model)

        vector_db_results = self.query_chroma_db(input_query, n_results, client=client, chroma_db=chroma_db)

        for i in vector_db_results:
            print(i)
            print("\n")
        
        text_results = ""
        len_results = 0

        for i in range(1, len(vector_db_results)):
            text_results += "-----------------------------------\n"
            text_results += "Result #"+str(i)+": \n"
            text_results += "Source: "+vector_db_results[i-1].metadata.get('source', 'Unknown')+"\n"
            text_results += "Page Number: "+str(vector_db_results[i-1].metadata.get('page', 'Unknown'))+"\n"
            text_results += vector_db_results[i-1].page_content
            text_results += "\n\n"

            len_results += len(vector_db_results[i-1].page_content)

        if self.is_debug:
            print(text_results)
        
        return text_results

    def fetch_vectors_postgres(self, connection_type, input_query, n_results=3):

        # Use query_postgres_db to query the PostgreSQL database
        vector_db_results = self.query_postgres_db(connection_type, input_query, n_results)

        for i in vector_db_results:
            print(i)
            print("\n")
        
        text_results = ""
        len_results = 0

        for i in range(len(vector_db_results)):
            print("Content: "+str(vector_db_results[i]))
            print("Datatype: "+ str(type(vector_db_results[i])))

            text_results += "-----------------------------------\n"
            text_results += "Result #"+str(i+1)+": \n"
            text_results += "Source: "+vector_db_results[i][1].get('source', 'Unknown')+"\n"
            text_results += "Page Number: "+str(vector_db_results[i][1].get('page', 'Unknown'))+"\n"
            text_results += vector_db_results[i][0]
            text_results += "\n\n"

            len_results += len(vector_db_results[i][0])

        if self.is_debug:
            print(text_results)
        
        return text_results

    def query_llm(self, sys_msg, human_msg, include_rag=True):
        # breakpoint()

        # print("what")
        if include_rag: 
            rag_results = self.fetch_vectors_postgres(self.connection_type, human_msg, 5)

            approx_token_length = len(rag_results) / 4

            if approx_token_length > 10000:
                print("Warning: Token length too long for ChatGPT rate limiting")
                print("Approx token length: "+str(approx_token_length)+" (Max length 10,000)")

                return
        
            full_message = "User question: "+human_msg+"\n\nSupporting materials: \n"+rag_results
        else:
            full_message = human_msg
        
        try:
            # Initialize ChatOpenAI with the instance API key
            model = langchain_openai.ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model_name="gpt-4",
                streaming=True,
                #temperature=0.7
            )
        except Exception as e:
            print(f"Error in model init: {str(e)}")
            raise
            
        message = [
                langchain_core.messages.SystemMessage(content=sys_msg),
                langchain_core.messages.HumanMessage(content=full_message)
            ]

            # Stream responses back from OpenAI
        response_generator = model.stream(message)
        print("\n\n///////////////CHATGPT RESPONSE////////////////////\n")

        for response in response_generator:
            if response.content:
                if self.is_debug:       
                    print(response)
                yield response.content

if __name__ == "__main__":

    input = "Please tell me about how I should organize my site into Areas of Concern."

    system_message_rag = """
    You are a helpful assistant who is aiding an environmental consultant to interpret New York City and New York State environmental regulation
    and its application to real estate construction projects. Your responses will be used to help write proposals for environmental site assessments.

    You will be fed a message from the consultant, as well as several pieces of supporting material that will be useful to you in answering the 
    consultant's question. The user's question will begin with "User question:", and will be followed up by two line breaks and a line marked 
    "Supporting materials:" that will mark the beginning of the supporting materials. Please use this information in your response as well as any
    other information that you have that may be beneficial in forming your response.
    
    """

    system_message_no_rag = """
    You are a helpful assistant who is aiding an environmental consultant to interpret New York City and New York State environmental regulation
    and its application to real estate construction projects. Your responses will be used to help write proposals for environmental site assessments.
    
    """

    llm_queryer = LLMQueryer(project_name='default')
    # breakpoint()
    llm_queryer.query_llm(sys_msg=system_message_rag, human_msg=input, include_rag=True)