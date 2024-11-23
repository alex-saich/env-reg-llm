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

if is_debug:
    ALLOW_RESET = os.getenv('ALLOW_RESET')
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
def process_and_store_pdfs(pdf_list, embedding_model=None, client=None, chroma_db=None, limit=None):
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    if client is None:
        client = chromadb.PersistentClient(path="vector_db")
    
    # Explicitly create the collection after deletion
    try:
        collection = client.create_collection("nyc_env_regs")
        if is_debug:
            print("Created new collection 'nyc_env_regs'")
    except ValueError:
        collection = client.get_collection("nyc_env_regs")
        if is_debug:
            print("Using existing collection 'nyc_env_regs'")

    chroma_db = Chroma(
        client=client,
        collection_name="nyc_env_regs", 
        embedding_function=embedding_model, 
        persist_directory="vector_db")

    j = 0
    for pdf in pdf_list:
        j+=1
        if is_debug:
            print("Starting split for " + os.path.basename(pdf)+"\n")

        chunks = load_and_chunk_pdf(pdf)

        i = 0
        # Extract text and metadata from chunks
        for chunk in chunks:
            i+=1

            if is_debug:       
                if i%5 == 0:
                    print("Inserting embed #"+str(i)+" of "+str(len(chunks))+"\n")

            text = chunk.page_content
            metadata = {"source": pdf, "page": chunk.metadata.get("page", "unknown"), "chunk": i}
            
            # Get all content for this PDF upfront
            if i == 1: # First chunk of this PDF
                pdf_content = chroma_db.get(
                    where={"source": {"$eq": pdf}}
                )
                existing_chunks = [meta["chunk"] for meta in pdf_content["metadatas"] 
                                if meta["source"] == pdf]


            # Only add if this chunk index doesn't already exist
            if i not in existing_chunks:
                # Store the embedding and metadata in ChromaDB
                chroma_db.add_texts([text], metadatas=[metadata])
            elif is_debug:
                print(f"Skipping chunk {i} from {pdf} as it already exists\n")
    
        if is_debug:
            print("Insert for "+os.path.basename(pdf)+" complete.\n")
        
        if j == limit:
            return

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

def delete_chroma_db(collection_name, persist_directory="vector_db"):
    """
    Deletes the specified Chroma collection and database.
    
    Args:
        collection_name (str): Name of collection to delete. Defaults to "nyc_env_regs"
        persist_directory (str): Path to vector database directory. Defaults to "vector_db"
    """
    try:
        # Initialize client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Delete the collection if it exists
        client.delete_collection(collection_name)
        
        # Reset the client connection
        if ALLOW_RESET:
            client.reset()
        
        if is_debug:
            print(f"Successfully deleted collection '{collection_name}' from {persist_directory}")
            
    except Exception as e:
        if is_debug:
            print(f"Error deleting database: {str(e)}")
        raise e


if __name__ == "__main__":
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    client = chromadb.PersistentClient(path="vector_db")
    
    delete_chroma_db("nyc_env_regs", "vector_db")
    
    pdf_list = [f"pdf_library/{f}" for f in os.listdir("pdf_library") if f.endswith('.pdf')]
    process_and_store_pdfs(pdf_list, embedding_model=embedding_model, client=client, limit=1)
    
    # Verify collection exists and has documents
    if is_debug:
        try:
            collection = client.get_collection("nyc_env_regs")
            count = collection.count()
            print(f"Collection exists with {count} documents")
        except Exception as e:
            print(f"Error verifying collection: {str(e)}")
    
    # Create a new Chroma instance after processing
    chroma_db = Chroma(
        client=client,
        collection_name="nyc_env_regs", 
        embedding_function=embedding_model, 
        persist_directory="vector_db")
    
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