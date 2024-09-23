import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt Template

prompt=ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant.Please respond to the queation asked"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title("Langchain Demo with Llama3.1")
input_text=st.text_input("What do you want to know?")

#Ollama Llama3.1 model
llm=OllamaLLM(model="llama3.1")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
