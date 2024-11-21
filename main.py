import streamlit as st
from query_llm import query_llm
import os

# Set up the Streamlit page
st.set_page_config(page_title="NYC Environmental Regulations Assistant", page_icon="🌱")

# Add a title and description
st.title("NYC Environmental Regulations Assistant")
st.write("Ask questions about NYC environmental regulations and get AI-powered answers.")

# Add tabs for different pages
tab1, tab2 = st.tabs(["Ask Questions", "Upload Documents"])

with tab1:
    # Create the input text area
    user_question = st.text_area("Enter your question:", height=100)

    # Create a submit button
    if st.button("Get Answer"):
        if user_question:
            # System message for the AI
            system_message = """
        You are a helpful assistant who is aiding an environmental consultant to interpret New York City and New York State environmental regulation
        and its application to real estate construction projects. Your responses will be used to help write proposals for environmental site assessments.

        You will be fed a message from the consultant, as well as several pieces of supporting material that will be useful to you in answering the 
        consultant's question. The user's question will begin with "User question:", and will be followed up by two line breaks and a line marked 
        "Supporting materials:" that will mark the beginning of the supporting materials. Please use this information in your response.

        For supporting materials, you will be provided with a title and page number of the document that the materials are sourced from. Please reference 
        these in your response when you leverage information directly from that supporting material. 
        
        """
            
            # Get response from LLM
            with st.spinner('Searching and generating response...'):
                response = query_llm(system_message, user_question, include_rag=True)
                
                # Display the response
                st.write("### Answer:")
                st.write(response)
        else:
            st.warning("Please enter a question first.")

    # Add a footer with instructions
    st.markdown("---")
    st.markdown("""
        **Tips for asking questions:**
        - Be specific about the regulations you're asking about
        - Include relevant context about your project
        - Ask one question at a time for best results
    """)

with tab2:
    st.header("Upload PDF Documents")
    st.write("Upload PDF documents to add to the knowledge base.")
    
    # Display existing PDFs
    st.subheader("Current PDF Library")
    pdf_files = [f for f in os.listdir("pdf_library") if f.endswith('.pdf')]
    if pdf_files:
        for pdf in pdf_files:
            st.text(pdf)
    else:
        st.text("No PDFs currently in library")
    
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner('Processing documents...'):
                # Save uploaded files temporarily
                pdf_paths = []
                for uploaded_file in uploaded_files:
                    with open(f"pdf_library/{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(f"pdf_library/{uploaded_file.name}")
                
                # Process and store PDFs
                from store_pdfs import process_and_store_pdfs
                process_and_store_pdfs(pdf_paths)
                
                st.success("Documents processed and added to the knowledge base!")
