from typing import Dict, List, Optional, Any
from yt_dlp import YoutubeDL
import pathlib
import json
import sys
import argparse
import shutil


class ytdl:

    def __init__(
        self,
        out_dir: Optional[pathlib.Path] = None,
        format: Optional[str] = "bv*+ba/b",
        merge_output_format: Optional[str] = None,
        quiet: Optional[bool] = True,
    ):

        if out_dir is None:

            self.out_dir = pathlib.Path.cwd() / "downloads"

        else:

            self.out_dir = out_dir

        opts = self.init_set_opts(
            out_dir=self.out_dir,
            format=format,
            merge_output_format=merge_output_format,
            quiet=quiet,
        )

        self.ydl = YoutubeDL(opts)

    def init_set_opts(self, **kwargs) -> Dict[str, Any]:

        self.opts = {
            "outtmpl": str(
                kwargs.get("out_dir", pathlib.Path.cwd())
                / "%(uploader)s - %(title)s.%(ext)s"
            ),
            "format": kwargs.get("format", "bv*+ba/b"),
            "restrictfilenames": True,
            "quiet": kwargs.get("quiet", True),
            "noprogress": kwargs.get("quiet", True),
            "nocheckcertificate": True,
        }

        if kwargs.get("merge_output_format") is not None:
            self.opts["merge_output_format"] = kwargs.get("merge_output_format", "mp4")
            print("Set merge_output_format to", self.opts["merge_output_format"])

        if shutil.which("ffmpeg") is None:
            fmt = self.opts.get("format", "")
            if "+" in fmt or "merge_output_format" in self.opts:
                print(
                    "Warning: ffmpeg not found in PATH. Falling back to single-file format 'best' to avoid merging.",
                    file=sys.stderr,
                )

                self.opts.pop("merge_output_format", None)
                self.opts["format"] = "best"

        return self.opts

    def get_info(self, url: str) -> Dict[str, Any]:

        opts = self.get_opts()

        with YoutubeDL(opts) as ydl:

            info = ydl.extract_info(url, download=True)

        return info

    def list_formats(self, url: str) -> List[Dict[str, Any]]:

        info = self.get_info(url)
        return info.get("formats", [])

    def download(
        self,
        url: str,
        subtitles: Optional[bool] = False,
        subtitleslangs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        opts = self.get_opts()

        if subtitles:

            self.opts.update(
                {
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": subtitleslangs or ["en"],
                }
            )

        with YoutubeDL(opts) as ydl:

            info = ydl.extract_info(url, download=True)
        return info

    def get_opts(self) -> Dict[str, Any]:

        return self.opts


if __name__ == "__main__":

    url = "https://www.youtube.com/watch?v=NMFPrCjpVCI"
    ytdl_instance = ytdl()

    info = ytdl_instance.get_info(url)
    print(info["title"], info["duration"], len(info["thumbnails"]))

    ytdl_instance.download(url)
