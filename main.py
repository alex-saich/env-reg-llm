import streamlit as st
from query_llm import LLMQueryer
from pull_db_data import DBManager
from store_pdfs_pg import PDFProcessor
import os
import copy

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
    project_options = DBManager(connection_type='streamlit').pull_project_names()
    
    # project = st.selectbox("Select Project", project_options)  # Default to first option if not found
    # st.session_state.project = project  # Update session state to the new selection
    
    project = project_options[0]
    # breakpoint()

    # Question input
    user_question = st.text_area("Enter your question:", height=100)

    system_message = st.text_area("System Message", value="""You are a helpful assistant who is aiding an environmental consultant to interpret New York City and New York State environmental regulation 
            and its application to real estate construction projects. Your responses will be used to help write proposals for environmental site assessments.

            You will be fed a message or set of messages from a conversation with the consultant, as well as several pieces of supporting material that will 
            be useful to you in answering the consultant's question. The latest user question will be followed by two line breaks and a line marked 
            "Supporting materials from database:" that will mark the beginning of the supporting materials. Please use this information in your response.

            For supporting materials, you will be provided with a title and page number of the document that the materials are sourced from. When relevant, please reference 
            these in your response when you leverage information directly from that supporting material. """, 
            help="These are admin instructions which are passed to the AI model. Update this to change the guidance you give on responses.", 
            label_visibility="visible" , height=300)

    # Submit button
    submit_button = st.button("Get Answer")

    # Create container for response after the button
    response_container = st.container()
    response_placeholder = response_container.empty()

    # Submit button
    if submit_button:
        if user_question:
   
            with response_container:
            # Clear the placeholder before starting
                response_placeholder.empty()
                full_response = ""

                list_user_q = user_question
                st.session_state.qa_history.insert(0,
                    {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": list_user_q
                            }
                        ]
                    }
                )

                message_history = copy.deepcopy(st.session_state.qa_history)
                
                llm_queryer = LLMQueryer(project_name=st.session_state.project,connection_type='streamlit')
                #with st.spinner('Searching and generating response...'):
                for chunk in llm_queryer.query_llm(sys_msg=system_message, human_msg=message_history,include_rag=True):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                # Final update without cursor
                response_placeholder.markdown(full_response)

                # Add to history after completion
                st.session_state.qa_history.insert(0,
                        {
                        "role": "assistant",
                        "content": [
                            {
                            "type": "text",
                            "text": full_response
                            }
                        ]
                        }
                )

        else:
            st.warning("Please enter a question first.")
    
    # Display Q&A history
    if st.session_state.qa_history:
        st.markdown("---")
        st.subheader("Full Chat History (Most Recent Texts First)")
        for qa in st.session_state.qa_history:
            if qa["role"] == "assistant":
                st.markdown("**Answer:**")
                st.markdown(qa["content"][0]["text"])
            elif qa["role"] == "user":
                st.markdown(":blue[Question:]")
                st.markdown(":blue[" + qa["content"][0]["text"] + "]")
            
with tab2:
           
    st.header("Upload PDF Documents")
    st.write("Upload PDF documents to add to the knowledge base.")
    
    # Create a dropdown for selecting project
    
    # st.subheader("Select Project")
    # project_options = pull_project_names() + ["Create New Project"]
    
    # # Use a selectbox with a callback to update session state
    # project = st.selectbox("Select Project", project_options, index=project_options.index(st.session_state.project) if 'project' in st.session_state else 0)
    
    # # Update session state directly after selection
    # st.session_state.project = project  # Ensure session state is updated here

    # if project == "Create New Project":
    #     new_project = st.text_input("Enter New Project Name")
    #     if st.button("Create Project"):
    #         if new_project not in project_options:
    #             message = insert_project_name(new_project)
    #             print(message)
    #             if "successfully" in message:
    #                 st.session_state.project = new_project
    #         else:
    #             st.warning("Project name already exists. Please choose a different name.")
    
    
    # Display existing PDFs for the selected project
    st.subheader("Current PDF Library")
    pdf_files = DBManager(connection_type='streamlit').pull_project_pdfs(project)
    print(pdf_files)
    if pdf_files:
        pdf_list = []  # Initialize a list to hold the rows
        for pdf in pdf_files:
            print(pdf)
            row = {"PDF Name": pdf[0], "Summary": pdf[1]}
            pdf_list.append(row)  # Append each row to the list
        
        st.table(pdf_list)  # Pass the list of rows to st.table
    else:
        st.text("No PDFs currently in library for this project")
    
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner('Processing documents...'):
                # Save uploaded files temporarily
                pdf_paths = []
                # Ensure the directory exists
                project_dir = f"pdf_library/{st.session_state.project}"
                os.makedirs(project_dir, exist_ok=True)  # Create the directory if it doesn't exist
                
                for uploaded_file in uploaded_files:
                    with open(f"{project_dir}/{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(f"{project_dir}/{uploaded_file.name}")
                
                # Process and store PDFs for the selected project
                processor = PDFProcessor(connection_type='streamlit',project=st.session_state.project)
                result = processor.process_and_store_pdfs(pdf_list=pdf_paths)
                
                if "successfully" in result:
                    st.success(result)
                else:
                    st.error(result)
