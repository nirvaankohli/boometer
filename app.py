# streamlit_app.py
from typing import List, Dict, Optional
from urllib.parse import parse_qs, urlparse
import math
from xml.parsers.expat import model
import streamlit as st
from youtubesearchpython import VideosSearch
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from PIL import Image
from classification.pretrained.api.inference.emotion.classification import Infer
from classification.pretrained.api.fear.scores.calculate import from_emotion_scores
from classification.preprocessing.image.cropping.face.transform import crop_face
from pathlib import Path


MODEL_PATH = (
    Path(__file__).parent
    / "classification"
    / "preprocessing"
    / "model"
    / "haarcascade_frontalface_default.xml"
)


def print_debug(*args):

    if debug:

        print_msg = ""

        for message in args:

            if type(message) == str:

                print_msg += message + " "

            else:

                print_msg += str(message) + " "

        print(f"[DEBUG] {print_msg.strip()}")


debug = True

print_debug("Initializing model...")
model = Infer()
print_debug("Model initialized.")


def get_youtube_id(url: str) -> Optional[str]:

    if not url:

        return None

    p = urlparse(url)
    if p.netloc in ("youtu.be", "www.youtu.be"):
        return p.path.strip("/")
    if "youtube.com" in p.netloc:
        return parse_qs(p.query).get("v", [None])[0]
    return None


def extract_needed_data(results: List[Dict], idx: int) -> list:

    if idx < 0 or idx >= len(results):
        return []

    vid = results[idx]

    thumbs = vid.get("thumbnails") or []
    thumb = thumbs[0]["url"] if thumbs else None
    title = vid.get("title") or "No Title"
    duration = vid.get("duration") or "N/A"
    views = (vid.get("viewCount") or {}).get("text", "N/A")
    published = vid.get("publishedTime") or "N/A"
    video_id = vid.get("id") or f"vid_{idx}"

    return [thumb, title, duration, views, published, video_id]


def youtube_results(query: str, limit: int) -> tuple[List[Dict], Optional[str]]:
    try:
        results = VideosSearch(query, limit=limit).result().get("result", [])
        if not results:
            return [], "No results found. Try a different search term."
        return results, None
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return [], "Search failed. Please try again in a moment."


st.set_page_config(page_title="YouTube Search & Watch", page_icon="▶", layout="wide")

if not hasattr(st.session_state, "results"):
    st.session_state.results = []
if not hasattr(st.session_state, "last_query"):
    st.session_state.last_query = ""
if not hasattr(st.session_state, "last_limit"):
    st.session_state.last_limit = 5


def centered(text: str, tag="p"):
    st.markdown(
        f"<{tag} style='text-align:center'>{text}</{tag}>", unsafe_allow_html=True
    )


qp = st.query_params

if "v" in qp and qp["v"]:

    token = qp["v"]
    vid = get_youtube_id(token) or token
    col1, col2 = st.columns([1, 1])
    video_url = f"https://www.youtube.com/watch?v={vid}"

    col1.markdown(
        f"""
        <div class="player">
            <iframe class="player ka-player-iframe centered-when-windowed _1rzb079g"
                name="ka-player-iframe"
                id="video_{vid}"
                title="YouTube video"
                frameborder="0"
                allowfullscreen
                src="https://www.youtube-nocookie.com/embed/{vid}/?controls=1&enablejsapi=1&modestbranding=1&showinfo=0&iv_load_policy=3&html5=1&fs=1&rel=0&hl=en"
                width="800"
                height="480"
                data-youtubeid="{vid}"
                data-translatedyoutubeid="{vid}"
                data-translatedyoutubelang="en"
                tabindex="0"
                allow="autoplay">
            </iframe>
        </div>
        <script src="https://www.youtube.com/iframe_api"></script>
        <script>
            function onYouTubeIframeAPIReady() {{
                new YT.Player('video_{vid}', {{ events: {{}} }});
            }}
        </script>
        <style>
            .player {{
                display: flex;
                width: 100%;
                justify-content: center;
                margin: 20px 0;
                
            }}

            .player iframe {{
                border-radius: 8px;
                overflow: hidden;
                border: 2px solid #ccc;
            }}
        </style>
        """,
        unsafe_allow_html=True,
        width="stretch",
    )

    with col2:

        class VideoProcessor(VideoTransformerBase):
            def __init__(self):
                self.face_detector = crop_face(MODEL_PATH, debug=debug)
                self.last_score = 0.0
                self.frame_count = 0
                self.process_this_frame = True

            def transform(self, frame):
                self.frame_count += 1
                img = frame.to_ndarray(format="bgr24")

                # Process every 3rd frame for performance
                if self.frame_count % 3 == 0:
                    try:
                        # Detect faces first
                        faces = self.face_detector.detect_faces(img)

                        if len(faces) > 0:
                            # Draw face rectangle
                            for x, y, w, h in faces:
                                cv2.rectangle(
                                    img, (x, y), (x + w, y + h), (0, 255, 0), 2
                                )

                            # Get the cropped face
                            img_proc = self.face_detector.crop_first_face(img)

                            if img_proc is not None and img_proc.size > 0:
                                # Ensure proper size and channels
                                if (
                                    len(img_proc.shape) == 3
                                ):  # Check if it's a valid color image
                                    # Convert to RGB for the model
                                    img_proc_rgb = cv2.cvtColor(
                                        img_proc, cv2.COLOR_BGR2RGB
                                    )
                                    img_proc_pil = Image.fromarray(img_proc_rgb)

                                    # Get emotion predictions
                                    model.set_image(img_proc_pil)
                                    results = model.predict()
                                    print_debug("Raw prediction results:", results)

                                    if (
                                        results
                                        and isinstance(results, list)
                                        and len(results) > 0
                                    ):
                                        scores = model.get_numeric_scores(results)
                                        print_debug("Numeric scores:", scores)

                                        # Validate scores before calculating fear
                                        if (
                                            scores
                                            and len(scores) > 0
                                            and all(
                                                isinstance(v, (int, float))
                                                for v in scores.values()
                                            )
                                        ):
                                            fear_calculator = from_emotion_scores(
                                                emotion_scores=scores, debug=True
                                            )
                                            new_score = (
                                                fear_calculator.calculate_fear_score()
                                            )

                                            if new_score is not None and not np.isnan(
                                                new_score
                                            ):
                                                self.last_score = new_score
                                                print_debug(
                                                    f"Updated fear score to: {self.last_score}"
                                                )

                    except Exception as e:
                        print_debug(f"Frame processing error: {str(e)}")
                        import traceback

                        print_debug("Traceback:", traceback.format_exc())

                # Always draw the score
                cv2.putText(
                    img,
                    f"Fear Score: {self.last_score:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                return img

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key=f"reaction_{vid}",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
        )

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
            "# of Results", 1, 15, int(st.session_state.last_limit), 1
        )
        go = st.button("Search", width="stretch")

    if go and query.strip():
        results, error = youtube_results(query.strip(), int(limit))
        if error:
            st.error(error)
        else:
            st.session_state.results = results
            st.session_state.last_query = query.strip()
            st.session_state.last_limit = int(limit)

    results = st.session_state.get("results", [])

    if results:

        num_columns = 2
        st_col = [1, 1] * num_columns
        num_rows = math.ceil(len(results) / num_columns)

        for i in range(num_rows):

            col = st.columns(st_col)

            base = i * num_columns

            for j in range(num_columns):

                idx = base + j

                if idx >= len(results):
                    break

                thumb, title, duration, views, published, video_id = (
                    extract_needed_data(results, idx)
                )

                left, right = col[j * 2], col[j * 2 + 1]
                with left:
                    if thumb:
                        st.image(thumb, width="stretch")
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
    elif st.session_state.get("last_query"):
        st.info("No results found. Try a different search term.")
