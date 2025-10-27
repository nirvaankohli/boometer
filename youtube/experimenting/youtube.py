from yt_dlp import YoutubeDL

# Options

ydl_opts = {
    "outtmpl": "%(uploader)s - %(title)s.%(ext)s",
    "format": "bv*+ba/b",         # best video+audio
    "merge_output_format": "mp4", # container if needed
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["en"],
    "quiet": True,
}
url = "https://www.youtube.com/watch?v=VIDEO_ID"
url = "https://www.youtube.com/watch?v=VIDEO_ID"

with YoutubeDL(ydl_opts) as ydl:
    
    info = ydl.extract_info(url, download=True)   
    print(info["title"], info["duration"], len(info["thumbnails"]))