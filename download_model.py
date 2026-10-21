"""Explicit download of Google's official MediaPipe hand landmark model."""

from pathlib import Path
from urllib.request import Request, urlopen

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEST = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MAX_BYTES = 30 * 1024 * 1024


def main() -> int:
    if DEST.exists():
        print(f"Model already exists: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".task.part")
    size = 0
    try:
        req = Request(MODEL_URL, headers={"User-Agent": "NV-AR-Portal-Educational/1.0"})
        with urlopen(req, timeout=35) as response, tmp.open("wb") as out:
            if response.url != MODEL_URL:
                raise RuntimeError("Unexpected redirect from official Google model URL")
            while chunk := response.read(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise RuntimeError("Model download exceeded size limit")
                out.write(chunk)
        if size < 100_000:
            raise RuntimeError("Downloaded model was unexpectedly small")
        tmp.replace(DEST)
        print(f"Model saved: {DEST} ({size:,} bytes)")
        return 0
    except Exception as error:
        print(f"Model download failed: {error}")
        return 1
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
