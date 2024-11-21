# Step 1: Install necessary packages
# pip install langchain chromadb pypdf openai

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

is_debug = os.getenv('DEBUG')

# Step 2: Set up API keys and initialize models
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # Set your OpenAI API key


# Step 3: Define function to load PDFs and split into chunks
def load_and_chunk_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Split text into manageable chunks (e.g., 500-1000 characters)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    # chunks = chunks[:20]

    if is_debug:
        print("   Split successful\n")

    if is_debug:
        print("Chunk Length: "+str(len(chunks))+"\n")
        print("First 3 entries:\n")
        print(str(chunks[:3])+"\n")
    return chunks

# Step 5: Process a list of PDFs and store in ChromaDB
def process_and_store_pdfs(pdf_list, embedding_model=None, client=None, chroma_db=None):
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    if client is None:
        client = chromadb.PersistentClient(path="vector_db")
    
    try:
        client.delete_collection("nyc_env_regs")
    except ValueError:
        pass  # Collection doesn't exist yet

    collection = client.create_collection("nyc_env_regs")

    chroma_db = Chroma(
        client=client,
        collection_name="nyc_env_regs", 
        embedding_function=embedding_model, 
        persist_directory="vector_db")

    for pdf in pdf_list:
        if is_debug:
            print("Starting split for " + os.path.basename(pdf)+"\n")

        chunks = load_and_chunk_pdf(pdf)

        i = 0
        # Extract text and metadata from chunks
        for chunk in chunks:
            if is_debug:
                i+=1
                if i%5 == 0:
                    print("Inserting embed #"+str(i)+" of "+str(len(chunks))+"\n")

            text = chunk.page_content
            metadata = {"source": pdf, "page": chunk.metadata.get("page", "unknown")}

            # Store the embedding and metadata in ChromaDB
            chroma_db.add_texts([text], metadatas=[metadata])
    
        if is_debug:
            print("Insert for "+os.path.basename(pdf)+" complete.\n")

# Example usage: Provide a list of PDFs to process


# Example query
# query_chroma_db("search term or question")

def grab_page_results_from_db(source_path, page_num, embedding_model=None, client=None, chroma_db=None):
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    if client is None:
        client = chromadb.PersistentClient(path="vector_db")
    if chroma_db is None:
        chroma_db = Chroma(collection_name="nyc_env_regs", embedding_function=embedding_model, persist_directory="vector_db")

    results = chroma_db.get(where={
                    "$and": [
                        {"source": {"$eq": source_path}},
                        {"page": {"$eq": page_num}}
                    ]
                    }
    )
    return results

if __name__ == "__main__":

    # Step 4: Initialize embedding model and ChromaDB vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    client = chromadb.PersistentClient(path="vector_db")
    chroma_db = Chroma(collection_name="nyc_env_regs", embedding_function=embedding_model, persist_directory="vector_db")
    
    pdf_list = [f"pdf_library/{f}" for f in os.listdir("pdf_library") if f.endswith('.pdf')]
    
    process_and_store_pdfs(pdf_list, embedding_model=embedding_model, client=client, chroma_db=chroma_db)

    # Retrieve first 3 documents from the collection
    # results = chroma_db.similarity_search("", k=20)

    
    source_path = "pdf_library/der-10.pdf"
    page_num = 1
    
    results = grab_page_results_from_db(source_path, page_num, embedding_model=embedding_model, client=client, chroma_db=chroma_db)

    print(results['documents'][0])
    

    """
    if is_debug:
        print("\nFirst 3 documents in ChromaDB:")
        for i, result in enumerate(results):
            print(f"\nDocument {i+1}:")
            print(f"Text: {result.page_content}\n")
            print(f"Metadata: {result.metadata}\n")
    """