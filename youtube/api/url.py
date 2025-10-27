from pytube import YouTube
import os
from typing import Any, List, Optional


class GetByURL:
    """Small helper around pytube.YouTube for selecting and downloading streams.

    Notes:
    - Initializes the YouTube object on construction.
    - Streams are lazily initialized (or can be refreshed with `init_streams`).
    """

    def __init__(self, url: str, init_streams: bool = False) -> None:
        self.url = url
        try:
            self.yt = YouTube(self.url)
        except Exception as e:  # keep broad to capture network/parse errors from pytube
            raise ValueError(f"Failed to create YouTube object for {url}: {e}")

        # stream query object (pytube.StreamQuery)
        self.yt_streams = self.yt.streams if init_streams else None
        self.stream = None

    def init_streams(self) -> None:
        """(Re)initialize the stream query object from the YouTube object."""
        self.yt_streams = self.yt.streams

    def get_yt_obj(self) -> YouTube:
        return self.yt

    def get_stream_query_object(self):
        return self.yt.streams

    def get_streams(self) -> List:
        """Return a list of available streams.

        Will initialize streams if they haven't been initialized yet.
        """
        if self.yt_streams is None:
            self.init_streams()

        # StreamQuery behaves like an iterable. Convert to list for callers.
        try:
            return list(self.yt_streams)
        except Exception:
            # older pytube had `.all()` on StreamQuery
            return self.yt_streams.all()

    def sort_streams(self, key: str) -> None:
        """Sort/order the internal stream query.

        Supported keys: "asc", "desc", "resolution" or any attribute accepted by `order_by`.
        """
        if self.yt_streams is None:
            self.init_streams()

        if key == "asc":
            # order by resolution ascending (best effort)
            self.yt_streams = self.yt_streams.order_by("resolution").asc()
        elif key == "desc":
            self.yt_streams = self.yt_streams.order_by("resolution").desc()
        elif key == "resolution":
            self.yt_streams = self.yt_streams.order_by("resolution")
        else:
            # pass-through to order_by with custom key
            self.yt_streams = self.yt_streams.order_by(key)

    def choose_stream_by(self, identifier: str, other_info: Optional[Any] = None):
        """Choose a stream and set `self.stream`.

        identifier can be: 'last', 'highest', 'lowest', 'itag', or defaults to 'first'.
        For 'itag', provide the itag in `other_info`.
        """
        if self.yt_streams is None:
            self.init_streams()

        if identifier == "last":
            self.stream = self.yt_streams.last()
        elif identifier == "highest":
            # pytube offers helper on the StreamQuery
            self.stream = self.yt_streams.get_highest_resolution()
        elif identifier == "lowest":
            self.stream = self.yt_streams.get_lowest_resolution()
        elif identifier == "itag":
            if other_info is None:
                raise ValueError(
                    "itag value must be provided in other_info when identifier == 'itag'"
                )
            self.stream = self.yt_streams.get_by_itag(int(other_info))
        else:
            self.stream = self.yt_streams.first()

        return self.stream

    def download(self, path: str = ".", filename: Optional[str] = None) -> str:
        """Download the currently chosen stream (or pick a sensible default) and return the full path.

        If no stream has been chosen, a progressive mp4 with highest resolution is chosen where possible.
        """
        if self.stream is None:
            # choose a common sensible default: progressive mp4 highest resolution
            self.stream = (
                self.yt.streams.filter(progressive=True, file_extension="mp4")
                .order_by("resolution")
                .desc()
                .first()
            )

            if self.stream is None:
                # fallback to the first available stream
                self.stream = self.yt.streams.first()

        if self.stream is None:
            raise RuntimeError("No stream available to download")

        # Ensure output directory exists
        abs_path = os.path.abspath(path)
        os.makedirs(abs_path, exist_ok=True)

        # Let pytube write the file. Then compute the full path returned.
        self.stream.download(output_path=abs_path, filename=filename)

        final_name = (
            filename if filename else getattr(self.stream, "default_filename", None)
        )
        if not final_name:
            # Very defensive fallback
            final_name = os.path.basename(self.yt.streams.first().default_filename)

        return os.path.join(abs_path, final_name)

    def download_by_order(
        self,
        path: str = ".",
        order: str = "first",
        filename: Optional[str] = None,
        other_info: Optional[Any] = None,
    ) -> str:
        """Download a stream chosen by `order` immediately and return the full path.

        order: 'first' (default), 'last', 'highest', 'lowest', 'itag'
        """
        streams = self.yt.streams

        if order == "last":
            stream = streams.last()
        elif order == "highest":
            stream = streams.get_highest_resolution()
        elif order == "lowest":
            stream = streams.get_lowest_resolution()
        elif order == "itag":
            if other_info is None:
                raise ValueError(
                    "itag must be provided via other_info when order == 'itag'"
                )
            stream = streams.get_by_itag(int(other_info))
        else:
            stream = streams.first()

        if stream is None:
            raise RuntimeError("Requested stream not found")

        abs_path = os.path.abspath(path)
        os.makedirs(abs_path, exist_ok=True)

        stream.download(output_path=abs_path, filename=filename)
        final_name = filename if filename else getattr(stream, "default_filename", None)
        if not final_name:
            final_name = os.path.basename(stream.default_filename)

        return os.path.join(abs_path, final_name)


if __name__ == "__main__":
    # Example usage
    video_url = "https://www.youtube.com/watch?v=9bZkp7q19f0"

    obj = GetByURL(video_url)
    obj.init_streams()
    obj.sort_streams("resolution")
    stream = obj.choose_stream_by("highest")
    download_path = obj.download(path=".", filename="downloaded_video.mp4")

    print(f"Video downloaded to: {download_path}")
