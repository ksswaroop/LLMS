from dotenv import load_dotenv
from IPython.display import display,Markdown
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

load_dotenv()

chat_model=ChatOpenAI(model_name="gpt-3.5-turbo")

def disp_markdown(text:str)-> None:
    display(Markdown(text))
embeddings=OpenAIEmbeddings()

with open("/home/swaroop/Documents/Coding/LLMS/Langchain/Build ChatGPT for Your Data/Alice_1.txt") as f:
    alice_in_wonderland=f.read()

#print(alice_in_wonderland)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator = "\n")

texts=text_splitter.split_text(alice_in_wonderland)
docsearch = Chroma.from_texts(texts, embeddings)

query = "What is the Rabbit late for?"

docs = docsearch.similarity_search(query,k=1)

print(docs[0])

chain=
