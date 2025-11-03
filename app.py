from typing import List, Dict, Optional
from urllib.parse import parse_qs, urlparse
import math
import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from youtubesearchpython import VideosSearch
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from PIL import Image
from classification.pretrained.api.inference.emotion.classification import Infer
from classification.pretrained.api.fear.scores.calculate import from_emotion_scores
from classification.preprocessing.image.cropping.face.transform import crop_face
from pathlib import Path
from collections import deque
import time
import queue
import csv
import os


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


def get_csv_path(vid: str) -> str:
    return "fear.csv"


def write_fear_data(vid: str, timestamp: float, fear_score: float, weighted_avg: float):
    csv_path = get_csv_path(vid)
    try:
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "fear_score", "weighted_avg"])

            writer.writerow([timestamp, fear_score, weighted_avg])
            f.flush()

        print_debug(
            f"CSV WRITTEN: {csv_path} - Fear: {fear_score:.2f}, WAvg: {weighted_avg:.2f}"
        )

    except Exception as e:
        print_debug("CSV write error:", str(e))


def read_latest_fear_data(vid: str) -> tuple:
    csv_path = get_csv_path(vid)
    print_debug(f"Reading from CSV: {csv_path}")
    try:
        if not os.path.exists(csv_path):
            print_debug(f"CSV file does not exist: {csv_path}")
            return None, None, None

        print_debug(f"CSV file exists, reading...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            print_debug(f"CSV has {len(rows)} rows")

            if len(rows) < 2:
                print_debug("No data rows found")
                return None, None, None

            last_row = rows[-1]
            print_debug(f"Last row: {last_row}")

            if len(last_row) >= 3:
                result = float(last_row[0]), float(last_row[1]), float(last_row[2])
                print_debug(f"Returning: {result}")
                return result

    except Exception as e:
        print_debug("CSV read error:", str(e))

    return None, None, None


def read_all_fear_data(vid: str) -> list:
    csv_path = get_csv_path(vid)
    data = []
    try:
        if not os.path.exists(csv_path):
            return data

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2:
                return data

            for row in rows[1:]:
                if len(row) >= 3:
                    timestamp = float(row[0])
                    dt = datetime.fromtimestamp(timestamp)
                    weighted_avg = float(row[2])
                    data.append({"t": dt, "weighted": weighted_avg})

    except Exception as e:
        print_debug("CSV read all error:", str(e))

    return data


@st.cache_resource(show_spinner="Loading emotion model (CPU)...")
def get_infer_model() -> Infer:
    print_debug("Initializing model (lazy)...")
    m = Infer()
    print_debug("Model initialized (lazy).")
    return m


def stop_webrtc_and_go_back(vid: str, ctx):

    try:
        if ctx is not None and hasattr(ctx, "stop"):
            ctx.stop()
            time.sleep(0.1)
    except Exception as e:
        print_debug("Error stopping WebRTC:", str(e))

    try:
        csv_path = get_csv_path(vid)
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print_debug(f"Cleaned up CSV file: {csv_path}")
    except Exception as e:
        print_debug("Error cleaning CSV:", str(e))

    chart_key = f"fear_chart_{vid}"
    if chart_key in st.session_state:
        st.session_state[chart_key] = None
    wavg_key = f"wavg_series_{vid}"
    if wavg_key in st.session_state:
        del st.session_state[wavg_key]
    if f"last_csv_time_{vid}" in st.session_state:
        del st.session_state[f"last_csv_time_{vid}"]

    try:
        del st.query_params["v"]
    except KeyError:
        pass
    st.rerun()


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

    plotly_placeholder = st.empty()
    status_placeholder = st.empty()

    with col1:
        
        st.markdown(
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

        proc_freq_key = f"proc_every_n_{vid}"
        st.slider(
            "Process every Nth frame (lower = more frequent)",
            min_value=1,
            max_value=5,
            value=1,
            key=proc_freq_key,
            help="Set to 1 to run classification on every frame. Increase to reduce CPU load.",
        )

        class VideoProcessor(VideoProcessorBase):

            def __init__(self):

                self.face_detector = crop_face(str(MODEL_PATH), debug=debug)
                self.model = get_infer_model()
                self.proc_freq_key = f"proc_every_n_{vid}"
                self.last_score = 0.0
                self.weighted_score = 0.0
                self.frame_count = 0
                self.process_this_frame = True
                self.scores = deque(maxlen=180)
                self.timestamps = deque(maxlen=180)
                self.vid = vid

            @staticmethod
            def weighted_average(scores) -> float:
                if not scores:
                    return 0.0
                arr = np.asarray(scores, dtype=float)
                arr = np.clip(arr, 0.0, 10.0)
                weights = np.maximum(arr, 1e-6)
                return float(np.sum(arr * weights) / np.sum(weights))

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                self.frame_count += 1
                img = frame.to_ndarray(format="bgr24")

                try:
                    proc_every = int(st.session_state.get(self.proc_freq_key, 1))
                except Exception:
                    proc_every = 1
                proc_every = max(1, proc_every)

                if self.frame_count % proc_every == 0:
                    try:
                        faces = self.face_detector.detect_faces(img)
                        if len(faces) > 0:
                            for x, y, w, h in faces:
                                cv2.rectangle(
                                    img, (x, y), (x + w, y + h), (0, 255, 0), 2
                                )

                            img_proc = self.face_detector.crop_first_face(img)
                            if (
                                img_proc is not None
                                and img_proc.size > 0
                                and len(img_proc.shape) == 3
                            ):
                                img_proc_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
                                img_proc_pil = Image.fromarray(img_proc_rgb)

                                self.model.set_image(img_proc_pil)
                                results = self.model.predict()
                                print_debug("Raw prediction results:", results)

                                if results and isinstance(results, list):
                                    scores = self.model.get_numeric_scores(results)
                                    print_debug("Numeric scores:", scores)

                                    if scores and all(
                                        isinstance(v, (int, float))
                                        for v in scores.values()
                                    ):
                                        new_score = from_emotion_scores(
                                            emotion_scores=scores, debug=True
                                        ).calculate_fear_score()
                                        if new_score is not None and not np.isnan(
                                            new_score
                                        ):

                                            self.last_score = new_score
                                            now = time.time()
                                            self.timestamps.append(now)
                                            self.scores.append(new_score)
                                            self.weighted_score = self.weighted_average(
                                                list(self.scores)
                                            )
                                            try:
                                                write_fear_data(
                                                    self.vid,
                                                    now,
                                                    self.last_score,
                                                    self.weighted_score,
                                                )
                                                print_debug(
                                                    f"CSV: Written fear data - Fear: {self.last_score:.2f}, WAvg: {self.weighted_score:.2f}"
                                                )
                                            except Exception as e:
                                                print_debug("CSV write error:", str(e))
                                            print_debug(
                                                f"Updated fear score to: {self.last_score} (weighted: {self.weighted_score:.2f})"
                                            )
                    except Exception as e:
                        import traceback

                        print_debug(f"Frame processing error: {str(e)}")
                        print_debug("Traceback:", traceback.format_exc())

                cv2.putText(
                    img,
                    f"Fear: {self.last_score:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    img,
                    f"WAvg: {self.weighted_score:.2f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        ctx = webrtc_streamer(
            key=f"reaction_{vid}",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.session_state[f"webrtc_ctx_{vid}"] = ctx

        chart_key = f"fear_chart_{vid}"
        if chart_key not in st.session_state:
            st.session_state[chart_key] = None

        wavg_key = f"wavg_series_{vid}"
        if wavg_key not in st.session_state:
            st.session_state[wavg_key] = []

        if ctx is not None:
            try:
                st.caption(f"WebRTC playing: {ctx.state.playing}")
                csv_timestamp, latest_fear, latest_wavg = read_latest_fear_data(vid)
                if csv_timestamp:
                    st.caption(
                        f"CSV data: Fear={latest_fear:.2f}, WAvg={latest_wavg:.2f}"
                    )
                    csv_path = get_csv_path(vid)
                    if os.path.exists(csv_path):
                        file_size = os.path.getsize(csv_path)
                        st.caption(f"CSV file: {csv_path} ({file_size} bytes)")
                else:
                    st.caption("No CSV data yet")
            except Exception:
                pass

        print_debug("Checking for CSV data...")

        csv_timestamp, latest_fear, latest_wavg = read_latest_fear_data(vid)
        print_debug(
            f"CSV read result: timestamp={csv_timestamp}, fear={latest_fear}, wavg={latest_wavg}"
        )

        if csv_timestamp:

            last_known_time = st.session_state.get(f"last_csv_time_{vid}", 0)
            print_debug(
                f"Comparing timestamps: new={csv_timestamp}, last={last_known_time}"
            )

            if csv_timestamp > last_known_time:
                print_debug("New data detected! Updating chart...")
                status_placeholder.caption(
                    f"Latest Fear: {latest_fear:.2f} • Weighted Avg: {latest_wavg:.2f}"
                )

                st.session_state[f"last_csv_time_{vid}"] = csv_timestamp

                time.sleep(0.1)
                st.rerun()
            else:
                print_debug("No new data, skipping update")
                status_placeholder.caption(
                    f"Latest Fear: {latest_fear:.2f} • Weighted Avg: {latest_wavg:.2f}"
                )
        else:
            print_debug("No CSV data found")

        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()

        current_time = time.time()
        if current_time - st.session_state.last_refresh > 2.0:
            st.session_state.last_refresh = current_time
            print_debug("Auto-refreshing to check for new data...")
            st.rerun()

        series = read_all_fear_data(vid)
        print_debug(f"Chart data points: {len(series)}")
        if len(series) > 0:
            times = [point["t"] for point in series]
            values = [point["weighted"] for point in series]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=values,
                    mode="lines+markers",
                    name="Weighted Fear",
                    line=dict(color="#ff6b6b", width=3),
                    marker=dict(size=4),
                )
            )

            fig.update_layout(
                title="Live Weighted Fear Score",
                xaxis_title="Time",
                yaxis_title="Weighted Fear (0-10)",
                yaxis=dict(range=[0, 10]),
                height=400,
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50),
            )

            plotly_placeholder.plotly_chart(fig, use_container_width=True)
        else:

            fig = go.Figure()
            fig.update_layout(
                title="Live Weighted Fear Score - Waiting for data...",
                xaxis_title="Time",
                yaxis_title="Weighted Fear (0-10)",
                yaxis=dict(range=[0, 10]),
                height=400,
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50),
            )
            plotly_placeholder.plotly_chart(fig, use_container_width=True)

        if st.button("← Back to Search"):

            existing_ctx = st.session_state.get(f"webrtc_ctx_{vid}")
            stop_webrtc_and_go_back(vid, existing_ctx)

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
