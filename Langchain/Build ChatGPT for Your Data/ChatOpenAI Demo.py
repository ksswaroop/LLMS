from dotenv import load_dotenv
from IPython.display import display,Markdown
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.messages.ai import AIMessage
load_dotenv()


def disp_markdown(text:str)-> None:
    display(Markdown(text))

chat_model=ChatOpenAI(model_name="gpt-3.5-turbo")

#System Message is Associated with the System role
system_message=SystemMessage(content="You are a food critic.")

#The Human Message is associated with the user role
user_message=HumanMessage(content="Do you think Kraft Dinner constitutes fine dinner?")

# The AI Message is associated with the assistant role
assistant_message=AIMessage(content="Egads! No, It most certainly does not!")

second_user_message=HumanMessage(content="What about Red Lobster, surely that is fine dining!")

#Create the list of prompts

list_of_prompts=[system_message,
                 user_message,
                 assistant_message,
                 second_user_message]

output=chat_model.invoke(list_of_prompts)

print(output)