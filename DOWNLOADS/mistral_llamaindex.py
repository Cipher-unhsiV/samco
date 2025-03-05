!pip install pypdf
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install sentence_transformers
!pip install llama-index==0.9.39

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt

!mkdir Data

documents = SimpleDirectoryReader("/content/Data/").load_data()

print(documents)

system_prompt = """You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your goal is to summarize pdf which may also include tabular columns, as
accurately as possible based on the instructions and context provided."""

query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

!huggingface-cli login

"""### Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1"""

import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=750,
    generate_kwargs={"temperature": 0.5, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

pip install -U langchain-community

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("Act as a professional share market analyzer. Provided a doc, summarize the whole pdf in bullet points and paragraphs (combined wherever neccessary) between 500 to 600 words. Keep it to the point and don't leave even a single information. Add all the major points such that the reader doesn't have to go through the doc never after reading the summary.")

query_engine = index.as_query_engine()
response = query_engine.query("""You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your task is to analyze the given document and generate a structured summary in approximately 500 words. Ensure the summary:

Captures all key points, including data, insights, and observations.
Clearly outlines the context, such as the purpose of the document and relevant background information.
Summarizes tabular data and numerical figures effectively, while retaining accuracy and relevance.
Highlights significant trends, comparisons, or impacts mentioned in the document.
Uses formal and precise language suitable for a corporate or academic audience.
The output should be well-organized with clear headings or bullet points where applicable. Avoid omitting any critical information, and focus on maintaining a balance between brevity and detail.""")

print(response)

