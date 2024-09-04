from dotenv import load_dotenv
from langchain_core.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from IPython.display import display,Markdown
from langchain_openai.chat_models.base import ChatOpenAI

load_dotenv()

def disp_markdown(text:str)-> None:
    display(Markdown(text))

chat_model=ChatOpenAI(model_name="gpt-3.5-turbo")

# We can signift variables we want access to by wrapping them in {}
system_prompt_template="You are an expert in {SUBJECT}, and you're currently feeling {MOOD}"
system_prompt_template=SystemMessagePromptTemplate.from_template(system_prompt_template)
user_prompt_template="{CONTENT}"
user_prompt_template=HumanMessagePromptTemplate.from_template(user_prompt_template)

# put them together into ChatPromptTemplate
chat_prompt=ChatPromptTemplate.from_messages([system_prompt_template,user_prompt_template])
# Note the method "to_messages()", that's what converts formatted prompt into 
formatted_chat_prompt= chat_prompt.format_prompt(SUBJECT="Sparkling waters",MOOD="Joyful",CONTENT="Is Bubly a good sparkling water?").to_messages()

#disp_markdown(chat_model.invoke(formatted_chat_prompt))
print(chat_model.invoke(formatted_chat_prompt).content)