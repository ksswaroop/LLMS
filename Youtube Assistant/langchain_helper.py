#from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
#from langchain-community import 
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings= OpenAIEmbeddings()
# video_url="https://youtu.be/XpC7SVDXimg?si=Ruw5UTSdlfXgG1Zv"
def create_vector_db_from_youtube_url(video_url:str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs,embeddings)
    return db

def get_response_from_query(db,query,k=3):
    # gpt-3.5-turbo-0125 can handle 16385 tokens and returns 4096
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = OpenAI(model_name="babbage-002")
    prompt = PromptTemplate(
        input_variables=["question","docs"],
        template="""You are a helpful YouTube assistant that that can answer questions about
        videos based on the video's transcript.
    Answer the following question:{question} 
    By searching the following video transcript: {docs}
    Only use the factual information from the transcript to answer the question.
    If you feel like you don't have enough information to answer the question,
    say "I don't know".
    Your answers should be detailed.
    """
    )
    chain=LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query,docs=docs_page_content)
    response = response.replace("\n","")
    return response,docs

