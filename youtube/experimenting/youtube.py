from yt_dlp import YoutubeDL

# Options

ydl_opts = {
    "outtmpl": "%(uploader)s - %(title)s.%(ext)s",
    "format": "bv*+ba/b",         # best video+audio
    "merge_output_format": "mp4", # container if needed
    "quiet": True,
}
url = "https://www.youtube.com/watch?v=NMFPrCjpVCI"

with YoutubeDL(ydl_opts) as ydl:

    info = ydl.extract_info(url, download=True)   
    print(info["title"], info["duration"], len(info["thumbnails"]))