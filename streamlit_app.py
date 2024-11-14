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

st.title("🎈 My new Streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
