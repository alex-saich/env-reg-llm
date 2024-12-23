from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
 
import os
import json
from pull_db_data import DBManager

class PDFSummarizer:
    def __init__(self, max_length=5000, chunk_size=5000, chunk_overlap=150):
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = OpenAIEmbeddings(
                    # openai_api_key=self.openai_api_key,
                    model="text-embedding-ada-002"
                )
        self.model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model_name="gpt-4",
                streaming=True,
                #temperature=0.7
            )

    def summarize_pdf(self, pdf_path):
        # Load the PDF file using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split the PDF content into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # Initialize the summaries
        short_summary = ""
        long_summary = ""

        # Process each chunk
        for chunk in chunks:
            # Extract the text from the chunk
            text = chunk.page_content

            # Tokenize the text
            inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=self.max_length, truncation=True)

            # Generate the short summary
            short_summary += self.model.generate(inputs, max_length=150, min_length=50, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)

            # Generate the long summary
            long_summary += self.model.generate(inputs, max_length=300, min_length=100, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)



        return short_summary, long_summary
