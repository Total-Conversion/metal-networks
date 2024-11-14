__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import time
import os
import shutil
import datetime
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader
import pandas
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from texttable import Texttable
import re


models = st.selectbox(
    "Models",
    (
        "Gemini 1.5 (itemization) + Claude 3.5 (inference)",
        "GPT4 (itemization) + GPT4 (inference)",
        "Gemini 1.5 (itemization) + Llama 3.1 (inference)",
        "GPT4 (itemization) + Llama 3.1 (inference)",
        "new line (itemization) + Claude 3.5 (inference)",
        "new line (itemization) + GPT4 (inference)",
        "new line (itemization) + Llama 3.1 (inference)"
    ),
    0,
)
output_format = st.selectbox("Models", ("table", "csv"), 0,)
output_in_single_table = st.checkbox('Output in single table', value=True)
remove_empty_columns_from_the_output = st.checkbox('Remove empty columns from the output', value=True)
output_with_additional_headers = st.checkbox('Output with additional headers', value=False)
catalogue_dropdown = st.selectbox("Catalog", ("Do not match to the catalog", "Do not match to the catalog"), 0,)
textarea = st.text_area('Query', '')
# button = st.button(label="Submit", on_click=aaa(log_area))
log_area = st.empty()

if 'decomposition_llm_gpt' not in st.session_state:
    st.session_state.decomposition_llm_gpt = False


# def aaa():
#     global product_catalog_retriever
#     global log_area
#     input = textarea
#     log_area.markdown(st.session_state.decomposition_llm_gpt)
#     st.session_state.decomposition_llm_gpt = not st.session_state.decomposition_llm_gpt

# st.button(label="Submit", on_click=aaa)

if st.button(label="Submit"):
    input = textarea
    log_area.markdown(st.session_state.decomposition_llm_gpt)
    st.session_state.decomposition_llm_gpt = ChatOpenAI(
        openai_api_key='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',
        openai_api_base='https://openrouter.ai/api/v1',
        model_name="google/gemini-flash-1.5",
        temperature=0
      )