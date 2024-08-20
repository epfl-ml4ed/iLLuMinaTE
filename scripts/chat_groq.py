# chat_groq.py

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate  # Ensure you import from the correct module
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

class chat_groq_model():

    def __init__(self, model_name, temperature, groq_key=None):
        self.model_name = model_name
        self.temperature = temperature
        self.conversation = None

        if groq_key==None:
                key=1
        else:
                key=groq_key
                
        self.groq_key = os.getenv(f'GROQ_KEY_{key}')

    def define_chat_model(self, conversation_template):
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.groq_key
        )

        print(llm)

        PROMPT = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=conversation_template
        )

        self.conversation = LLMChain(
            prompt=PROMPT,
            llm=llm,
            memory=memory,
            verbose=False
        )
        return self.conversation

    def call(self, prompt):
        llm_response = self.conversation(input=prompt)
        return llm_response
