import os
import streamlit as st 
import pickle
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_helper import main_lang
from langchain_community.vectorstores import FAISS
from uuid import uuid4

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# we want llm to be precise
llm = OllamaFunctions(model="llama3.2", temperature=0, format="json")

query="What is the fiscal deficit of govt w.r.t govt's target ?. Give the output in correct format pls."

dim=768
index = faiss.IndexFlatL2(dim)
# load recent data
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


if not os.path.isdir("faiss_index"):
    chunks = main_lang(URL="https://pulse.zerodha.com/",max_loadable_links=25)
    #add document chunks
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)
    vector_store.save_local("faiss_index")
else:
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

langchain.debug=True
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
result = chain({"question": query}, return_only_outputs=True)
print(result)