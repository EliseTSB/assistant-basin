"""Portable audio decoder used by the copied fitting script.

The inputs are the ESC-50, Pixabay chicken, and VocalImitationSet audio files.
Pass their directories to weight_edit.py; ffmpeg is resolved from PATH unless
an executable path is supplied explicitly.
"""

import subprocess
from pathlib import Path

import numpy as np


def decode(path, *, ffmpeg="ffmpeg"):
    command = [
        str(ffmpeg), "-nostdin", "-loglevel", "error", "-i", str(Path(path)),
        "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(
        command, capture_output=True, timeout=60, check=True
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()
