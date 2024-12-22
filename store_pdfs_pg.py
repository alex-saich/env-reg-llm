from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import streamlit as st
import os
import json
from pull_db_data import DBManager


load_dotenv()

is_debug = os.getenv('DEBUG')

if is_debug:
    ALLOW_RESET = os.getenv('ALLOW_RESET')


def load_and_chunk_pdf(file_path):
    """
    Loads a PDF file and splits its content into chunks.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        list: A list of chunked PDF content.
    """
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the PDF content into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    if is_debug:
        print("   Split successful\n")

    if is_debug:
        print("Chunk Length: "+str(len(chunks))+"\n")
        print("First entry:\n")
        print(str(chunks[0])+"\n")
    return chunks

def process_and_store_pdfs(pdf_list, connection_type, project, embedding_model=None, limit=None):
    """
    Processes and stores PDFs in the database with their embeddings.
    
    Args:
        pdf_list (list): List of PDF file paths.
        embedding_model (OpenAIEmbeddings, optional): The embedding model to use. Defaults to None.
        limit (int, optional): Limit the number of PDFs to process. Defaults to None.
    
    Returns:
        str: Error message if any database operation fails, otherwise a success message.
    """
    # Set the default embedding model if not provided
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Establish a connection to the database
    try:
        conn = DBManager(connection_type='streamlit').get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        return f"Error connecting to database: {str(e)}"

    # Process each PDF in the list
    j = 0
    for pdf in pdf_list:
        j+=1
        if is_debug:
            print("Starting split for " + os.path.basename(pdf)+"\n")

        # Load and chunk the PDF
        chunks = load_and_chunk_pdf(pdf)

        # Process each chunk
        i = 0
        for chunk in chunks:
            i+=1

            if is_debug:       
                if i%5 == 0:
                    print("Inserting embed #"+str(i)+" of "+str(len(chunks))+"\n")

            # Extract the text and metadata from the chunk
            text = chunk.page_content
            metadata = {"source": os.path.basename(pdf), "page": chunk.metadata.get("page", "unknown"), "chunk": i}
            
            # Check if the chunk already exists in the database
            cur.execute(
                "SELECT COUNT(*) FROM doc_vectors WHERE document_name = %s AND document_chunk_id = %s",
                (metadata["source"], i)
            )
            if cur.fetchone()[0] == 0:
                # Embed the text using the OpenAI model
                embedding = embedding_model.embed_query(text)
                # Insert the chunk into the database
                try:
                    cur.execute(
                        "INSERT INTO doc_vectors (document_name, document_chunk_id, document_page_number, chunk_text, metadata, embedding, project_name) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (metadata["source"], i, metadata["page"], text, json.dumps(metadata), embedding, project)
                    )
                    conn.commit()
                except Exception as e:
                    return f"Error inserting chunk into database: {str(e)}"

        if is_debug:
            print("Insert for "+os.path.basename(pdf)+" complete.\n")

        # Check if the project already exists in the database, if not insert
        cur.execute(
            "SELECT COUNT(*) FROM projects WHERE project_name = %s",
            (metadata["source"],)
        )
        if cur.fetchone()[0] == 0:
            try:
                cur.execute(
                    "INSERT INTO documents (project_name, document_name) VALUES (%s, %s)",
                    (project, metadata["source"],)
                )
                conn.commit()
            except Exception as e:
                return f"Error inserting project into database: {str(e)}"

        # Stop processing if the limit is reached
        if j == limit:
            break

    # Close the database connection
    cur.close()
    conn.close()

    return("The pdf(s) were successfully inserted into the database.")

def grab_page_results_from_db(source, page_num):
    """
    Retrieves page results from the database for a given source and page number.
    
    Args:
        source (str): The source PDF name.
        page_num (int): The page number to retrieve results for.
    
    Returns:
        list: A list of tuples containing the results.
    """
    # Establish a connection to the database
    try:
        conn = DBManager(connection_type='streamlit').get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        if is_debug:
            print(f"Error connecting to database: {str(e)}")
        raise e

    # Retrieve the page results from the database
    cur.execute(
        "SELECT * FROM doc_vectors WHERE document_name = %s AND document_page_number = %s",
        (source, page_num)
    )
    results = cur.fetchall()

    # Close the database connection
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
    
    # Establish a connection to the database
    try:
        conn = DBManager(connection_type='streamlit').get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        if is_debug:
            print(f"Error connecting to database: {str(e)}")
        raise e

    # Delete the data from the database
    cur.execute(
        "DELETE FROM doc_vectors WHERE document_name LIKE %s",
        (f"%{source}%/",)
    )
    conn.commit()

    # Close the database connection
    cur.close()
    conn.close()

    if is_debug:
        print(f"Successfully deleted data for source '{source}' from PostgreSQL database.")

if __name__ == "__main__":
    # Set the embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Establish a connection to the database
    try:
        conn = DBManager(connection_type='streamlit').get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        if is_debug:
            print(f"Error connecting to database: {str(e)}")
        raise e
    
    # project = 'default'
    
    # Get the list of PDF files
    pdf_list = [f"pdf_library/{f}" for f in os.listdir("pdf_library") if f.endswith('.pdf')]
    # Process and store the PDFs
    process_and_store_pdfs(pdf_list, embedding_model=embedding_model, limit=1)
    
    # Verify data exists in PostgreSQL database
    if is_debug:
        try:
            cur.execute("SELECT COUNT(*) FROM doc_vectors")
            count = cur.fetchone()[0]
            print(f"Data exists with {count} records")
        except Exception as e:
            print(f"Error verifying data: {str(e)}")
    
    # Retrieve page results from the database
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