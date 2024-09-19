import streamlit as st

from src.ui_util import set_header, build_page, build_sidebar       # type: ignore
from src.model_util import InitChat                                 # type: ignore
from src.retriever_util import InitRetriever                        # type: ignore

def render_ui():
    """render ui/ux"""

    # build page config
    build_page()
    # set header
    set_header()
    # set page sidebar
    build_sidebar()

@st.cache_resource
def load_model(model_id: str, keep_alive: str, max_tokens: int, temperature: float):
    """load model"""
    
    llm = InitChat(model_id, keep_alive, max_tokens, temperature)
    return llm

@st.cache_resource
def load_retriever(device, metadata_uri, milvus_db_uri, collection_name):
    """filter context"""

    retriever = InitRetriever(device, metadata_uri, milvus_db_uri, collection_name)
    return retriever

def format_history_for_prompt(messages):
    """prepare chat history for llm"""

    prompt = ""
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        prompt += f"{role}: {message['content']}\n"
    return prompt

if __name__ == "__main__":
    # device cpu | cuda
    device = "cuda"

    # model google/gemma-2-2b-it | google/gemma-2-9b-it | gemma2:2b
    model_id = "gemma2:2b"

    # data source
    metadata_uri = "data/processed_data/metadata.csv"
    milvus_db_uri = "data/db/milvus_db/zalando_fashionista.db"
    collection_name = "zalando_fashionista_collection"

    # initialize streamlit session state 
    if "messages" not in st.session_state:
        greeting = "Hello! I'm your Zalando sales assistant. How can I help you today?"
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

    # render ui/ux
    render_ui()

    # load model
    keep_alive = "3h"
    max_tokens = 512
    temperature = 0
    llm = load_model(model_id, keep_alive, max_tokens, temperature)

    # load retriever
    retriever = load_retriever(device, metadata_uri, milvus_db_uri, collection_name)

    # talk to bot
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Say something"):        
        chat_history = format_history_for_prompt(st.session_state.messages)

        # user message
        with st.chat_message("user"):
            st.write(query)
            st.session_state.messages.append({"role": "user", "content": query})

        context = retriever.search(query)
        temp_context = context[['sku', 'description']]

        # convert filtered DataFrame to JSON string
        json_str_temp_context = temp_context.to_json(orient='records')

        print("query: ", query)
        print("context: ", json_str_temp_context)
        print("chat_history: ", chat_history)
        
        response = llm.talk_to_bot(query, json_str_temp_context, chat_history)
        print("response: ", type(response))
        print("response: ", response)
        
        with st.chat_message("assistant"):
            st.write(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
    