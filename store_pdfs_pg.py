from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain_core.documents
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import langchain_core
from langchain_core import documents
from dotenv import load_dotenv
import streamlit as st
import os
import json
from pull_db_data import DBManager


class PDFProcessor:
    def __init__(self, connection_type='streamlit', project='default'):
        self.connection_type = connection_type
        self.is_debug = os.getenv('DEBUG')
        self.db_manager = DBManager(connection_type=self.connection_type)
        self.project = project
        
        try: 
            self.model = ChatOpenAI(
                    openai_api_key=os.environ.get("OPENAI_API_KEY"),
                    model_name="gpt-4",
                )
        except Exception as e:
            print(f"Error in model init: {str(e)}")
            raise

        try: 
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        except Exception as e:
            print(f"Error in embedding model init: {str(e)}")
            raise

    def load_and_chunk_pdf(self, file_path, chunk_size=5000, chunk_overlap=150):
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        converted_chunks = []
        for chunk in chunks:
            page_content = chunk.page_content
            page_number = chunk.metadata.get("page")
            converted_chunks.append(
                {'page_content': page_content, 'page_number': page_number}
                )

        if self.is_debug:
            print("   Split successful\n")

        if self.is_debug:
            print("Chunk Length: "+str(len(converted_chunks))+"\n")
            print("First entry:\n")
            print(str(converted_chunks[0])+"\n")

        return converted_chunks
    
    def summarize_text(self, text_name, chunks, _depth=0):        
        
        print("Input Type: "+str(type(chunks)))

        orig_text_length = 0

        if len(chunks) > 1:

            i = 1

            new_chunks = []
            new_page = ""
            page_num = 1

            for chunk in chunks:

                chunk_len = len(chunk['page_content'])
                
                print("Chunk #"+str(i)+": "+str(chunk_len))
                message = [
                        langchain_core.messages.SystemMessage(content=""),
                        langchain_core.messages.HumanMessage(content=f"Summarize the following text in 1 to 3 paragraphs: {chunk['page_content']}")
                    ]

                summary_response = self.model.invoke(message).content

                if len(summary_response) + len(new_page) > 10000:
                    new_chunks.append(
                                    {'page_content': new_page, 'page_number': page_num}
                                    )
                    new_page = summary_response
                    page_num += 1
                else:
                    new_page += summary_response
                    new_page += '\n\n'
                
                i+=1

                if i == len(chunks):
                    new_chunks.append(
                        {'page_content': new_page, 'page_number': page_num}
                    )
                
                orig_text_length += chunk_len
            
            _depth +=1

            print('This Round Total Length: '+str(orig_text_length))
            print('\n')

            return self.summarize_text(text_name, new_chunks, _depth)

        print("Final pass: "+str(len(chunks[0]['page_content'])))

        long_summary = chunks[0]['page_content']

        message = [
            langchain_core.messages.SystemMessage(content=""),
            langchain_core.messages.HumanMessage(content=f"Summarize the following text into 1 to 2 sentences: {long_summary}")
        ]
        one_sentence_summary = self.model.invoke(message).content

        if _depth == 0:
            message = [
                    langchain_core.messages.SystemMessage(content=""),
                    langchain_core.messages.HumanMessage(content=f"Summarize the following text in 1 to 2 paragraphs: {long_summary}")
                ]

            long_summary = self.model.invoke(message).content
        
        long_summary = f"The following is a summary of the document \"{text_name}\": {long_summary}"
        long_summary_embedding = self.embedding_model.embed_query(long_summary)
        
        print('FINAL TEXT LENGTH: '+str(len(long_summary)))

        return one_sentence_summary, long_summary, long_summary_embedding

    def process_and_store_pdfs(self, pdf_list, limit=None):
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

        
        # Establish a connection to the database
        try:
            conn = self.db_manager.get_db_connection()
            cur = conn.cursor()
        except Exception as e:
            return f"Error connecting to database: {str(e)}"
        
        # insert the project_name into db if it doesn't already exist
        try:
            cur.execute(
                "INSERT INTO projects (project_name) VALUES (%s) ON CONFLICT (project_name) DO NOTHING",
                (self.project,)
            )
            conn.commit()
        except Exception as e:
            return f"Error inserting project into projects table: {str(e)}"

        # Process each PDF in the list
        pdf_limit = 0
        for pdf in pdf_list:

            pdf_limit+=1

            if self.is_debug:
                print("Starting split for " + os.path.basename(pdf)+"\n")

            chunk_id = 0
            pdf_name = os.path.basename(pdf)
            summary_chunks = self.load_and_chunk_pdf(pdf,chunk_size=10000)
                
                # call summarize_text
            one_sentence_summary, summary, summary_embedding = self.summarize_text(text_name=pdf_name,chunks=summary_chunks) 
            metadata = {"source": pdf_name, "page": 0, "chunk": chunk_id}

            try:
                cur.execute(
                    "INSERT INTO doc_vectors (document_name, document_chunk_id, document_page_number, chunk_text, metadata, embedding, project_name) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (document_name, document_chunk_id) DO NOTHING",
                    (metadata["source"], chunk_id, metadata["page"], summary, json.dumps(metadata), summary_embedding, self.project)
                )
                conn.commit()
            except Exception as e:
                return f"Error inserting summary into doc_vectors table: {str(e)}"
            

            print(type(metadata["source"]))
            print(type(one_sentence_summary))
            print(type(self.project))
            try:
                cur.execute(
                    "INSERT INTO documents (document_name, document_summary, project_name) VALUES (%s, %s, %s) ON CONFLICT (document_name) DO UPDATE SET document_summary = EXCLUDED.document_summary;",
                    (metadata["source"], one_sentence_summary, self.project)
                )
                conn.commit()
            except Exception as e:
                return f"Error inserting summary into documents table: {str(e)}"
            
            print("Onto the next")
            
            rag_chunks = self.load_and_chunk_pdf(pdf,chunk_size=5000)

            # Process each chunk
            for chunk in rag_chunks:
                chunk_id+=1

                if self.is_debug:       
                    if chunk_id%5 == 0:
                        print("Inserting embed #"+str(chunk_id)+" of "+str(len(rag_chunks))+"\n")

                # Extract the text and metadata from the chunk
                text = chunk['page_content']
                metadata = {"source": pdf_name, "page": chunk['page_number'], "chunk": chunk_id}
                
                # Check if the chunk already exists in the database
                cur.execute(
                    "SELECT COUNT(*) FROM doc_vectors WHERE document_name = %s AND document_chunk_id = %s",
                    (metadata["source"], chunk_id)
                )
                if cur.fetchone()[0] == 0:
                    # Embed the text using the OpenAI model
                    embedding = self.embedding_model.embed_query(text)
                    # Insert the chunk into the database
                    try:
                        cur.execute(
                            "INSERT INTO doc_vectors (document_name, document_chunk_id, document_page_number, chunk_text, metadata, embedding, project_name) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (metadata["source"], chunk_id, metadata["page"], text, json.dumps(metadata), embedding, self.project)
                        )
                        conn.commit()
                    except Exception as e:
                        return f"Error inserting chunk into database: {str(e)}"

            if self.is_debug:
                print("Insert for "+pdf_name+" complete.\n")

            # Stop processing if the limit is reached
            if pdf_limit == limit:
                break

        # Close the database connection
        cur.close()
        conn.close()

        return("The pdf(s) were successfully inserted into the database.")

    def grab_page_results_from_db(self, source, page_num):
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
            conn = self.db_manager.get_db_connection()
            cur = conn.cursor()
        except Exception as e:
            if self.is_debug:
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

    def delete_from_postgres_db(self, source):
        """
        Deletes the specified source pdf's data from the PostgreSQL database.
        
        Args:
            source (str): Name of source to delete data for.
        """
        
        # Establish a connection to the database
        try:
            conn = self.db_manager.get_db_connection()
            cur = conn.cursor()
        except Exception as e:
            if self.is_debug:
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

        if self.is_debug:
            print(f"Successfully deleted data for source '{source}' from PostgreSQL database.")

if __name__ == "__main__":

    # load_dotenv()

    # is_debug = os.getenv('DEBUG')

    # if is_debug:
    #     ALLOW_RESET = os.getenv('ALLOW_RESET')
    
    # # Set the embedding model    
    # # Establish a connection to the database
    # try:
    #     conn = DBManager(connection_type='streamlit').get_db_connection()
    #     cur = conn.cursor()
    # except Exception as e:
    #     if is_debug:
    #         print(f"Error connecting to database: {str(e)}")
    #     raise e
    
    # # project = 'default'
    # processor = PDFProcessor(connection_type='local')
    
    # # Get the list of PDF files
    # pdf_list = [f"pdf_library/{f}" for f in os.listdir("pdf_library") if f.endswith('.pdf')]
    # # Process and store the PDFs
    # processor.process_and_store_pdfs(pdf_list, limit=1)
    
    # # Verify data exists in PostgreSQL database
    # if is_debug:
    #     try:
    #         cur.execute("SELECT COUNT(*) FROM doc_vectors")
    #         count = cur.fetchone()[0]
    #         print(f"Data exists with {count} records")
    #     except Exception as e:
    #         print(f"Error verifying data: {str(e)}")
    
    # # Retrieve page results from the database
    # source_path = "der-10.pdf"
    # page_num = 1
    
    # results = processor.grab_page_results_from_db(source_path, page_num)

    # print(results)

    """
    if is_debug:
        print("\nFirst 3 documents in PostgreSQL database:")
        for i, result in enumerate(results):
            print(f"\nDocument {i+1}:")
            print(f"Text: {result[0]}\n")
            print(f"Metadata: {result[1]}, {result[2]}, {result[3]}\n")
    """

    tester = PDFProcessor(connection_type='local')
    test_file = tester.process_and_store_pdfs(['pdf_library/default/Track order status - Grailed - 1ZAC78610317381697.pdf'])
    print(test_file)

    # print("One Sentence Summary: "+one_sentence+"\n\n")
    # print("Full Summary: "+summary)




