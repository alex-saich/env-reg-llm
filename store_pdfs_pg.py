from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import streamlit as st
import os
import json
import psycopg2

load_dotenv()

is_debug = os.getenv('DEBUG')

if is_debug:
    ALLOW_RESET = os.getenv('ALLOW_RESET')

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

def load_and_chunk_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    if is_debug:
        print("   Split successful\n")

    if is_debug:
        print("Chunk Length: "+str(len(chunks))+"\n")
        print("First entry:\n")
        print(str(chunks[0])+"\n")
    return chunks

def process_and_store_pdfs(pdf_list, embedding_model=None, limit=None):
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        if is_debug:
            print(f"Error connecting to database: {str(e)}")
        raise e

    j = 0
    for pdf in pdf_list:
        j+=1
        if is_debug:
            print("Starting split for " + os.path.basename(pdf)+"\n")

        chunks = load_and_chunk_pdf(pdf)

        i = 0
        for chunk in chunks:
            i+=1

            if is_debug:       
                if i%5 == 0:
                    print("Inserting embed #"+str(i)+" of "+str(len(chunks))+"\n")

            text = chunk.page_content
            metadata = {"source": os.path.basename(pdf), "page": chunk.metadata.get("page", "unknown"), "chunk": i}
            
            cur.execute(
                "SELECT COUNT(*) FROM doc_vectors WHERE document_name = %s AND document_chunk_id = %s",
                (metadata["source"], i)
            )
            if cur.fetchone()[0] == 0:
                embedding = embedding_model.embed_query(text)
                cur.execute(
                    "INSERT INTO doc_vectors (document_name, document_chunk_id, document_page_number, chunk_text, metadata, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                    (metadata["source"], i, metadata["page"], text, json.dumps(metadata), embedding)
                )
                conn.commit()

        if is_debug:
            print("Insert for "+os.path.basename(pdf)+" complete.\n")

        # Check if project already exists, if not insert
        cur.execute(
            "SELECT COUNT(*) FROM projects WHERE project_name = %s",
            (metadata["source"],)
        )
        if cur.fetchone()[0] == 0:
            cur.execute(
                "INSERT INTO projects (project_name) VALUES (%s)",
                (metadata["source"],)
            )
            conn.commit()

        if j == limit:
            break

    cur.close()
    conn.close()

def grab_page_results_from_db(source, page_num):
    
    # Connect to PostgreSQL database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        if is_debug:
            print(f"Error connecting to database: {str(e)}")
        raise e

    cur.execute(
        "SELECT * FROM doc_vectors WHERE document_name = %s AND document_page_number = %s",
        (source, page_num)
    )
    results = cur.fetchall()

    cur.close()
    conn.close()
    return results

def delete_from_postgres_db(source, db_config=None):
    """
    Deletes the specified source pdf's data from the PostgreSQL database.
    
    Args:
        source (str): Name of source to delete data for.
        db_config (dict): Database configuration. Defaults to None.
    """
    
    # Connect to PostgreSQL database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        if is_debug:
            print(f"Error connecting to database: {str(e)}")
        raise e

    cur.execute(
        "DELETE FROM doc_vectors WHERE document_name LIKE %s",
        (f"%{source}%/",)
    )
    conn.commit()

    cur.close()
    conn.close()

    if is_debug:
        print(f"Successfully deleted data for source '{source}' from PostgreSQL database.")

if __name__ == "__main__":
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        if is_debug:
            print(f"Error connecting to database: {str(e)}")
        raise e
    
    # project = 'default'
    
    pdf_list = [f"pdf_library/{f}" for f in os.listdir("pdf_library") if f.endswith('.pdf')]
    process_and_store_pdfs(pdf_list, embedding_model=embedding_model, limit=1)
    
    # Verify data exists in PostgreSQL database
    if is_debug:
        try:
            cur.execute("SELECT COUNT(*) FROM doc_vectors")
            count = cur.fetchone()[0]
            print(f"Data exists with {count} records")
        except Exception as e:
            print(f"Error verifying data: {str(e)}")
    
    source_path = "der-10.pdf"
    page_num = 1
    
    results = grab_page_results_from_db(source_path, page_num)

    print(results)

    """
    if is_debug:
        print("\nFirst 3 documents in PostgreSQL database:")
        for i, result in enumerate(results):
            print(f"\nDocument {i+1}:")
            print(f"Text: {result[0]}\n")
            print(f"Metadata: {result[1]}, {result[2]}, {result[3]}\n")
    """