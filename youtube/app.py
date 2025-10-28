import math
from pdb import run
from typing import List, Dict, Optional
import streamlit as st
from youtubesearchpython import VideosSearch

st.set_page_config(
    page_title="(Demo & Test) Youtube Search & Download",
    page_icon="▶",
    layout="wide",
)


# Things to get vids


def search(query: str, limit: int, page: int = 1) -> List[Dict]:

    offset = (page - 1) * limit

    vs = VideosSearch(query, limit=limit)

    return vs.result().get("result", [])


# get search results
@st.cache_data
def get_search_results(query: str, limit: int, page: int) -> List[Dict]:

    return search(query, limit, page)


def centered_text(text: str, heading: str = "p"):

    st.markdown(
        f"<{heading} style='text-align: center;'>{text}</{heading}>",
        unsafe_allow_html=True,
    )


center_col = st.columns([1, 6, 1])[1]

with center_col:

    centered_text("Youtube Search & Download", "h1")
    centered_text("Search and watch Youtube videos directly here!", "h4")
    query = st.text_input("Search Youtube Videos", value="", max_chars=100)
    st.
