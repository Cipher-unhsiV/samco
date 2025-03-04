import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext
import os
import torch
from PyPDF2 import PdfReader

# Set up the Streamlit app
st.title("Share Market PDF Summarizer")
st.write("Upload a PDF file to get a concise, professional summary of its content.")

# PDF Upload
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file
    pdf_path = "uploaded_file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from the PDF
    st.write("Processing the PDF file...")
    reader = PdfReader(pdf_path)
    text_data = ""
    for page in reader.pages:
        text_data += page.extract_text()
    
    # Save the text data to a directory for processing
    os.makedirs("Data", exist_ok=True)
    with open("Data/doc.txt", "w") as f:
        f.write(text_data)
    
    # Set up the LLM and embeddings
    system_prompt = """You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your goal is to summarize pdf which may also include tabular columns, as accurately as possible based on the instructions and context provided."""
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=750,
        generate_kwargs={"temperature": 0.5, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        device_map="auto",
        #model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    # Load the document into the index
    documents = SimpleDirectoryReader("Data").load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # Generate the summary
    query_engine = index.as_query_engine()
    query = """You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your task is to analyze the given document and generate a structured summary in approximately 500 words. Ensure the summary:
    
    Captures all key points, including data, insights, and observations.
    Clearly outlines the context, such as the purpose of the document and relevant background information.
    Summarizes tabular data and numerical figures effectively, while retaining accuracy and relevance.
    Highlights significant trends, comparisons, or impacts mentioned in the document.
    Uses formal and precise language suitable for a corporate or academic audience.
    The output should be well-organized with clear headings or bullet points where applicable. Avoid omitting any critical information, and focus on maintaining a balance between brevity and detail."""
    
    response = query_engine.query(query)
    
    # Display the summary
    st.subheader("Summary:")
    st.write(response)

    # Clean up temporary files
    os.remove(pdf_path)
    os.remove("Data/doc.txt")
    os.rmdir("Data")
