import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space  # type: ignore

def set_header():
    """set page header"""
    st.header("Zalando Bot: AI-Powered Fashion Assistant", divider="rainbow")

def build_page():
    """build page ui/ux"""

    st.set_page_config(
        page_title="Zalando Query Bot: AI-Powered Fashion Assistant",
        page_icon="🛠️",
        layout="centered",
        initial_sidebar_state="expanded"
    )  

def build_sidebar():
    """build sidebar"""

    img_path = "./docs/img/query_bot_logo.png"
    st.sidebar.image(img_path, use_column_width=True)

    add_vertical_space(2)
    
    st.sidebar.markdown(
        """
        ### Zalando Query Bot
        Built with:    
        * [Streamlit](https://streamlit.io)
        * [Google Gemma2 9B](https://huggingface.co/google/gemma-2-9b-it)
        * [Milvus Vector Database](https://milvus.io)
        ___
        """
    )    