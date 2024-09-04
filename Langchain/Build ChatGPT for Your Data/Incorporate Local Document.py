from dotenv import load_dotenv
from IPython.display import display,Markdown
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
load_dotenv()

chat_model=ChatOpenAI(model_name="gpt-3.5-turbo")

def disp_markdown(text:str)-> None:
    display(Markdown(text))
embeddings=OpenAIEmbeddings()

with open("/home/swaroop/Documents/Coding/LLMS/Langchain/Build ChatGPT for Your Data/Alice_1.txt") as f:
    alice_in_wonderland=f.read()

print(alice_in_wonderland)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator = "\n")

texts=text_splitter.split_text(alice_in_wonderland)
docsearch = Chroma.from_texts(texts, embeddings)

query = "What is the Rabbit late for?"

docs = docsearch.similarity_search(query,k=1)

#########

# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# K = 3  


# def create_embedding_model():
#     """Creates an instance of the OpenAI embedding model."""
#     return OpenAIEmbeddings()

# file_path="/home/swaroop/Documents/Coding/LLMS/Langchain/Build ChatGPT for Your Data/Alice_1.txt")

# def create_vector_database(file_path):
#     """
#     Creates a FAISS vector database from the provided PDF reader object.

#     Args:
#         pdf_reader (PyPDF2.PdfReader): The PDF reader object containing the document content.

#     Returns:
#         FAISS: The created FAISS vector database.
#     """
#     with open(file_path)as f:
#         texts=f.read()


#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         length_function=len
#     )

#     chunks = text_splitter.split_text(texts)
#     db = Chroma.from_texts(chunks, embedding=create_embedding_model())
#     return db

