from dotenv import load_dotenv
from IPython.display import display,Markdown
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
load_dotenv()


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K = 3  


# def disp_markdown(text:str)-> None:
#     display(Markdown(text))

def create_embedding_model():
    """Creates an instance of the OpenAI embedding model."""
    return OpenAIEmbeddings()

chat_model=ChatOpenAI(model_name="gpt-3.5-turbo")


def create_vector_database(file_path):
    with open(file_path) as f:
        alice_in_wonderland=f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len)
    texts=text_splitter.split_text(alice_in_wonderland)
    #print(len(texts))
    chunks = text_splitter.split_text(texts)
    db = Chroma.from_texts(chunks, embedding=create_embedding_model())
    return db

