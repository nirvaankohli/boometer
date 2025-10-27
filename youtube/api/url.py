from typing import Dict, List, Optional, Any
from yt_dlp import YoutubeDL
import pathlib
import json
import sys
import argparse


class ytdl:

    def __init__(
        self,
        out_dir: Optional[pathlib.Path] = None,
        format: Optional[str] = "bv*+ba/b",
        merge_output_format: Optional[str] = "mp4",
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
            "merge_output_format": kwargs.get("merge_output_format", "mp4"),
            "restrictfilenames": True,
            "merge_output_format": kwargs.get("merge_output_format", "mp4"),
            "quiet": kwargs.get("quiet", True),
            "noprogress": kwargs.get("quiet", True),
            "nocheckcertificate": True,
        }

        return self.opts
    
    def get_info

    def get_opts(self):

        return self.opts
