"""Compatibility shim for the removed stdlib imghdr module in Python 3.13+.

Streamlit 1.19 imports imghdr to detect image types. This module reintroduces a
minimal imghdr.what implementation using Pillow so the app can run.
"""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO, Optional, Union

from PIL import Image

_ImageSource = Union[str, bytes, bytearray, BinaryIO]


def what(file: _ImageSource, h: Optional[bytes] = None) -> Optional[str]:
    """Return the image type based on filename or data stream.

    Args:
        file: A filename, bytes-like object, or file-like object.
        h: Optional bytes to identify instead of reading from file.
    """
    try:
        if h is not None:
            stream = BytesIO(h)
            with Image.open(stream) as img:
                return _normalize_format(img.format)

        if isinstance(file, (bytes, bytearray)):
            stream = BytesIO(file)
            with Image.open(stream) as img:
                return _normalize_format(img.format)

        if hasattr(file, "read"):
            data = file.read()
            stream = BytesIO(data)
            with Image.open(stream) as img:
                return _normalize_format(img.format)

        with Image.open(file) as img:
            return _normalize_format(img.format)
    except Exception:
        return None


def _normalize_format(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    return fmt.lower()
