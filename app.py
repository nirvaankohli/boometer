# streamlit_app.py
from typing import List, Dict, Optional
from urllib.parse import parse_qs, urlparse

import streamlit as st
from youtubesearchpython import VideosSearch


def get_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    p = urlparse(url)
    if p.netloc in ("youtu.be", "www.youtu.be"):
        return p.path.strip("/")
    if "youtube.com" in p.netloc:
        return parse_qs(p.query).get("v", [None])[0]
    return None


def youtube_results(query: str, limit: int) -> List[Dict]:
    return VideosSearch(query, limit=limit).result().get("result", [])


st.set_page_config(page_title="YouTube Search & Watch", page_icon="▶", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_limit" not in st.session_state:
    st.session_state.last_limit = 5


def centered(text: str, tag="p"):
    st.markdown(
        f"<{tag} style='text-align:center'>{text}</{tag}>", unsafe_allow_html=True
    )


qp = st.query_params

if "v" in qp and qp["v"]:

    token = qp["v"]
    vid = get_youtube_id(token) or token

    st.video(f"https://www.youtube.com/watch?v={vid}")

    if st.button("← Back to Search"):

        try:

            del st.query_params["v"]

        except KeyError:

            pass

        st.rerun()

# Search page
else:

    center = st.columns([1, 6, 1])[1]

    with center:

        centered("YouTube Search & Watch", "h1")
        centered("Search and watch YouTube videos here.", "h4")

        query = st.text_input(
            "Search YouTube Videos", value=st.session_state.last_query, max_chars=100
        )
        limit = st.number_input(
            "# of Results", 1, 50, int(st.session_state.last_limit), 1
        )
        go = st.button("Search")

    if go and query.strip():

        st.session_state.results = youtube_results(query.strip(), int(limit))
        st.session_state.last_query = query.strip()
        st.session_state.last_limit = int(limit)

    results = st.session_state.results

    if results:

        for idx, vid in enumerate(results):

            col = st.columns([1, 4])

            left, right = st.columns([1, 4])
            thumbs = vid.get("thumbnails") or []
            thumb = thumbs[0]["url"] if thumbs else None
            title = vid.get("title") or "No Title"
            duration = vid.get("duration") or "N/A"
            views = (vid.get("viewCount") or {}).get("text", "N/A")
            published = vid.get("publishedTime") or "N/A"
            video_id = vid.get("id") or f"vid_{idx}"

            with left:
                if thumb:
                    st.image(thumb, use_container_width=True)
                else:
                    st.write("No thumbnail")

            with right:
                st.markdown(f"### {title}")
                st.caption(
                    f"Duration: {duration} • Views: {views} • Published: {published}"
                )
                if st.button("Watch", key=f"watch_{video_id}"):

                    st.query_params["v"] = video_id
                    st.rerun()

            st.markdown("---")
    elif st.session_state.last_query:
        st.info("No results.")
