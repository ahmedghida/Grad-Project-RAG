import streamlit as st
from Scripts import *
import os

# Initialize session state for storing vectorstore and chat history
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar for uploading PDF
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Display the uploaded file name
    st.sidebar.write(f"Uploaded file: {uploaded_file.name}")
    
    # Process PDF and create a vectorstore
    documents = get_pdf_text(uploaded_file)
    file_name = os.path.splitext(uploaded_file.name)[0]
    
    # Check if vectorstore already exists, if not, create it
    if not st.session_state["vectorstore"]:
        st.session_state["vectorstore"] = create_vectorstore_from_texts(documents, file_name)
        st.sidebar.success("Vectorstore created successfully!")

# Main chat interface
st.title("Chat with your PDF")

# Display previous messages in the chat
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from user (chat query)
if user_query := st.chat_input("Ask a question about the PDF"):
    # Add user message to the chat history
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Check if the vectorstore exists
    if st.session_state["vectorstore"]:
        # Fetch the answer from the vectorstore using the query_document function
        response = query_document(st.session_state["vectorstore"], user_query)
        
        # Add the system's response to the chat history
        st.session_state["messages"].append({"role": "system", "content": response})
        
        # Display system's response
        with st.chat_message("system"):
            st.markdown(response)
    else:
        st.warning("Please upload a PDF first.")
