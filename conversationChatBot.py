# to reference environment variables
from dotenv import load_dotenv
import os
# to use chat model
from langchain.chat_models import AzureChatOpenAI
# establish conversation memory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# to set up prompt template
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
# for UX
import streamlit as st

# load env variables
load_dotenv()

st.title("Azure Chat Open AI conversation")

# moving memory, chat_model, prompt, conversation, session state initialization outside of main.
# Because streamlit loops over main repeatedly. 

# set up chat memory with specific key variable
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# set up chat model
chat_model = AzureChatOpenAI(deployment_name = 'gpt-4')

# define prompt
prompt = ChatPromptTemplate(
messages=[
    SystemMessagePromptTemplate.from_template(
        "You are a nice chatbot knowledgable about everything. \
        You are helping and in a conversation with a human."
    ),
    # The `variable_name` here is what must align with memory
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}") # start here: check if 2 variables can be added here: question and context
]
)




# setting up chain
conversation = LLMChain(
    llm=chat_model,
    prompt=prompt,
    verbose=False,
    memory=memory
    )
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def main(memory):
    print('Hello')
    
    
    # Concatenate all messages in chat history to create context
    context = {
        'context': "\n".join([message["content"] for message in st.session_state.messages])
    }
    
    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    
    
    # Accept user inpupt
    if prompt := st.chat_input("Talk to AzureChatOpen AI gpt-4"):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)          
        
        # Add user message to chat history
        st.session_state.messages.append({'role':'Human', 'content': prompt})
    
        # Display assistant response in chat message container        
        with st.chat_message('assistant'):
            llm_response = conversation.run(question=prompt)# , context=context)
            st.markdown(llm_response)

            # Add assistant response to chat history
            st.session_state.messages.append({'role':'AIMessage', 'content': llm_response})
    




if __name__ == '__main__': main(memory)
