# Grad-Project-RAG
## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to interact with PDFs through a chat interface. Leveraging Llama 3 (70B) and Google’s `text-embedding-001` model via the Gemini key, the system extracts text from uploaded PDFs, processes it into chunks, embeds these chunks, and stores them in Chroma DB for efficient retrieval.

## Features

- **Chat Interface**: Users can upload PDFs and chat with the content.
- **Text Extraction**: Converts PDF documents into text.
- **Chunking and Embedding**: Splits text into manageable chunks and embeds them for efficient retrieval.
- **Containerized Deployment**: The project is packaged in a Docker container for easy deployment.

## Requirements

- Docker
- Python 3.x
- Streamlit
- Chroma DB
- Access to Llama 3 API and Google `text-embedding-001`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedghida/Grad-Project-RAG
   cd Grad-Project-RAG
   ``` 

2. Build the Docker image:
   ``` bash
    Copy code
    docker build -t rag-app .
    ```

## Running the Docker Container

To run the Docker container, follow these steps:

```bash
docker run -p 8501:8501 rag-app
```
## Usage
    Upload your PDF file through the provided interface.
    Start chatting with the content of the PDF.
    The system will utilize the RAG model to generate responses based on the embedded text.

## Project Structure
``` bash
Grad-Project-RAG
│
├── Dockerfile           # Dockerfile for containerization
├── app.py               # Main Streamlit application file
├── requirements.txt     # Python dependencies
└── ...                  # Additional files and directories

```
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- `Llama 3` and Google `text-embedding-001` for their powerful NLP capabilities.
- Chroma DB for efficient data storage and retrieval.
