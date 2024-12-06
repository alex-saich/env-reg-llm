import streamlit as st
from query_llm import query_llm
from pull_db_data import pull_project_names
import os

# Set up the Streamlit page
st.set_page_config(page_title="NYC Environmental Regulations Assistant", page_icon="🌱")

# Add a title and description
st.title("NYC Environmental Regulations Assistant")
st.write("Ask questions about NYC environmental regulations and get AI-powered answers.")

# Add tabs for different pages
tab1, tab2 = st.tabs(["Ask Questions", "Upload Documents"])

with tab1:
    # Initialize session states
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'project' not in st.session_state:
        st.session_state.project = "default"

    # Tips section
    st.markdown("""
        **Tips for asking questions:**
        - Be specific about the regulations you're asking about
        - Include relevant context about your project
        - Ask one question at a time for best results
    """)

    # Project selection
    project_options = pull_project_names()
    project = st.selectbox("Select Project", project_options, index=project_options.index(st.session_state.project))
    st.session_state.project = project

      # Question input
    user_question = st.text_area("Enter your question:", height=100)

    # Submit button
    submit_button = st.button("Get Answer")

    # Create container for response after the button
    response_container = st.container()
    response_placeholder = response_container.empty()

    # Submit button
    if submit_button:
        if user_question:

            system_message = """
            You are a helpful assistant who is aiding an environmental consultant to interpret New York City and New York State environmental regulation
            and its application to real estate construction projects. Your responses will be used to help write proposals for environmental site assessments.

            You will be fed a message from the consultant, as well as several pieces of supporting material that will be useful to you in answering the 
            consultant's question. The user's question will begin with "User question:", and will be followed up by two line breaks and a line marked 
            "Supporting materials:" that will mark the beginning of the supporting materials. Please use this information in your response.

            For supporting materials, you will be provided with a title and page number of the document that the materials are sourced from. Please reference 
            these in your response when you leverage information directly from that supporting material. 
            """
            
            with response_container:
            # Clear the placeholder before starting
                response_placeholder.empty()
                full_response = ""

                #with st.spinner('Searching and generating response...'):
                for chunk in query_llm(sys_msg=system_message, human_msg=user_question):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                # Final update without cursor
                response_placeholder.markdown(full_response)

                # Add to history after completion
                st.session_state.qa_history.insert(0, {
                    "question": user_question,
                    "answer": full_response
                })
        else:
            st.warning("Please enter a question first.")

    # Display Q&A history
    if st.session_state.qa_history:
        st.markdown("---")
        st.subheader("Full Chat History (Most Recent Texts First)")
        for qa in st.session_state.qa_history:
            st.markdown("**Answer:**")
            st.markdown(qa["answer"])
            st.markdown(":blue[Question:]")
            st.write(":blue[" + qa["question"] + "]")
            
with tab2:
           
    st.header("Upload PDF Documents")
    st.write("Upload PDF documents to add to the knowledge base.")
    
    # Create a dropdown for selecting project
    st.subheader("Select Project")
    project_options = [f for f in os.listdir("pdf_library") if os.path.isdir(os.path.join("pdf_library", f))] + ["Create New Project"]
    project = st.selectbox("Select Project", project_options)
    if project == "Create New Project":
        new_project = st.text_input("Enter New Project Name")
        if st.button("Create Project"):
            os.makedirs(f"pdf_library/{new_project}", exist_ok=True)
            st.session_state.project = new_project
    else:
        st.session_state.project = project
    
    # Display existing PDFs for the selected project
    st.subheader("Current PDF Library")
    pdf_files = [f for f in os.listdir(f"pdf_library/{st.session_state.project}") if f.endswith('.pdf')]
    if pdf_files:
        for pdf in pdf_files:
            st.text(pdf)
    else:
        st.text("No PDFs currently in library for this project")
    
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner('Processing documents...'):
                # Save uploaded files temporarily
                pdf_paths = []
                for uploaded_file in uploaded_files:
                    with open(f"pdf_library/{st.session_state.project}/{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(f"pdf_library/{st.session_state.project}/{uploaded_file.name}")
                
                # Process and store PDFs for the selected project
                from store_pdfs_pg import process_and_store_pdfs
                process_and_store_pdfs(pdf_paths, project=st.session_state.project)
                
                st.success("Documents processed and added to the knowledge base for the selected project!")
