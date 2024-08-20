from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate  # Ensure you import from the correct module
from langchain_groq import ChatGroq
from langchain_community.llms import Replicate
from getpass import getpass

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

class chat_replicate_model():

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.conversation = None
        self.replicate_key = os.getenv

    def define_chat_model(self, conversation_template):
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = Replicate(
            model=self.model_name,
            model_kwargs={"temperature": self.temperature},
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
