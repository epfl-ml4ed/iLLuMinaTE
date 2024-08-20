from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate

class chat_openai_model():
    def __init__(self, model_name, temperature):
        self.model_name=model_name
        self.temperature = temperature
        self.conversation = None

    def define_chat_model(self, conversation_template):
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
        )

        PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=conversation_template)

        self.conversation = LLMChain(
            prompt=PROMPT,
            llm=llm,
            memory = memory,
            verbose=False
        )
        return self.conversation

    def call(self, prompt):
        llm_response = self.conversation(input=prompt)
        return llm_response