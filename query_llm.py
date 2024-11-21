from dotenv import load_dotenv
import langchain_openai 
import langchain_core
import langchain_chroma 
import chromadb
import os

load_dotenv()

is_debug = os.getenv('DEBUG')

# Step 2: Set up API keys and initialize models
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # Set your OpenAI API key

def query_chroma_db(query, n_results=3, embedding_model=None, client=None, chroma_db=None):
    if embedding_model is None:
        embedding_model = langchain_openai.OpenAIEmbeddings(model="text-embedding-ada-002")
    if client is None:
        client = chromadb.PersistentClient(path="vector_db")
    if chroma_db is None:
        chroma_db = langchain_chroma.Chroma(collection_name="nyc_env_regs", embedding_function=embedding_model, persist_directory="vector_db")

    results = chroma_db.similarity_search(query, n_results)
    
    if is_debug:
        print(str(type(results)))
        print(str(len(results)))

    return results

def fetch_vectors(input_query):

    embedding_model = langchain_openai.OpenAIEmbeddings(model="text-embedding-ada-002")
    client = chromadb.PersistentClient(path="vector_db")
    chroma_db = langchain_chroma.Chroma(client=client, collection_name="nyc_env_regs", embedding_function=embedding_model)

    vector_db_results = query_chroma_db(input_query,3,embedding_model=embedding_model,client=client,chroma_db=chroma_db)

    for i in vector_db_results:
        print(i)
        print("\n")
    
    text_results = ""
    len_results = 0

    for i in range(1,len(vector_db_results)):
        text_results += "-----------------------------------\n"
        text_results += "Result #"+str(i)+": \n"
        text_results += "Source: "+vector_db_results[i-1].get('source', 'Unknown')+"\n"
        text_results += "Page Number: "+vector_db_results[i-1].get('page', 'Unknown')+"\n"
        text_results += vector_db_results[i-1].page_content
        text_results += "\n\n"

        len_results += len(vector_db_results[i-1].page_content)

    if is_debug:
        print(text_results)
    
    return text_results

def query_llm(sys_msg,human_msg,include_rag=True):

    if include_rag: 
        rag_results = fetch_vectors(human_msg)

        approx_token_length = len(rag_results) / 4

        if approx_token_length > 10000:
            print("Warning: Token length too long for ChatGPT rate limiting")
            print("Approx token length: "+str(approx_token_length)+" (Max length 10,000)")

            return
        
        full_message = "User question: "+human_msg+"\n\nSupporting materials: \n"+rag_results
    else:
        full_message = human_msg
    
    model = langchain_openai.ChatOpenAI(model="gpt-4")
    parser = langchain_core.output_parsers.StrOutputParser()

    message = [
        langchain_core.messages.SystemMessage(content=sys_msg),
        langchain_core.messages.HumanMessage(content=full_message)
    ]

    """
    response = model.invoke(message)
    print("API response: ")
    print(response)
    print("\n")

    
    text_only = parser.invoke(response)
    print("Message parser: ")
    print(text_only)
    """  

    chain = model | parser
    response = chain.invoke(message)
    
    if is_debug:
        print("\n\n///////////////CHATGPT RESPONSE////////////////////\n")
        print(response)

    
    return response

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
    query_llm(system_message_no_rag, input,include_rag=False)