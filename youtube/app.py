import math
from typing import List, Dict, Optional
import streamlit as st
from youtubesearchpython import VideosSearch

st.set_page_config(
    page_title="(Demo & Test) Youtube Search & Download",
    page_icon="▶",
    layout="wide",
)

st.title("Youtube Search & Download (Demo & Test)")
st.caption("Just a test & demo for boometer")

# Things to get vids


def search(query: str, limit: int, page: int = 1) -> List[Dict]:

    offset = (page - 1) * limit

    vs = VideosSearch(query, limit=limit, offset=offset)

    return vs.result().get("result", [])


# get search results
@st.cache_data
def get_search_results(query: str, limit: int, page: int) -> List[Dict]:

    return search(query, limit, page)


# sidebar for like customization and inputs

with st.sidebar:

    # All the inputs
    st.header("Search Parameters")
    query = st.text_input("Search Query", value="boometer")
    per_page = st.slider("Results per page", min_value=1, max_value=20, value=5)
    page = st.number_input(
        "Page Number", min_value=1, max_value=50, value=1, step=1
    )
    run = st.button("Search", type="primary")

if run & (query.strip() != ""):

    items = get_search_results(query, per_page, int(page))

    if not items:

        st.warning("No results found. Try different query or page number.")

    else:

        cols_per_row = 3 
        rows = math.ceil(len(items) / cols_per_row) # get rows

        for i in range(rows):

            cols = st.columns(cols_per_row)

            for col, item in zip(cols, items[i * cols_per_row : (i + 1) * cols_per_row]):

                with col:

                    thumbnails = item.get("thumbnails", [])
                    thumb = thumbnails[-1]["url"] if thumbnails else None
                    
                    if thumb:
                        st.image(thumb, use_column_width=True)
                    
                    st.markdown(f"**{item.get('title', 'No Title')}**")
                    st.markdown(f"Channel: {item.get('channel', {}).get('name', 'Unknown')}")
                    st.markdown(f"Duration: {item.get('duration', 'N/A')}")
                    st.markdown(f"Views: {item.get('viewCount', {}).get('text', 'N/A')}")
                    st.markdown(f"[Watch on Youtube](https://www.youtube.com/watch?v={item.get('id', '')})")
                    
                    video_id = item.get("id", None)
                    url = f"https://www.youtube.com/watch?v={video_id}"  if video_id else None
                    
                    watch = st.button("Watch")

                    if watch and url:

                        st.video(url)

else:

    st.write("Enter a search query and click 'Search' to find Youtube videos.")
