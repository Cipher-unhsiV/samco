# -*- coding: utf-8 -*-
"""Mistral_with_llamaindex.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MI46WHjhzqN6lmQX817q7nzm9_3labjs
"""

!nvidia-smi

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

system_prompt = """You are a pdf summarizer assistant. Your goal is to summarize pdf which may also include tabular columns, as
accurately as possible based on the instructions and context provided."""

query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

!huggingface-cli login

"""### Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1"""

import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
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
response = query_engine.query("Summarize the whole pdf in bullet points. keep it to the point and don't leave even a single information.")

print(response)

