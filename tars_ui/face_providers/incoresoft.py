import os, requests, cv2, numpy as np
from .base import FaceProvider, FaceResult

class IncoresoftFaceProvider(FaceProvider):
    def __init__(self, base_url: str = None, api_key: str = None, timeout: float = 6.0):
        self.base_url = (base_url or os.getenv("INCORESOFT_BASE_URL","")).rstrip("/")
        self.api_key = api_key or os.getenv("INCORESOFT_API_KEY","")
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise RuntimeError("INCORESOFT_BASE_URL / INCORESOFT_API_KEY not set")

        self.identify_path = "/faces/identify"

    def _jpg(self, frame_bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok: raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def identify(self, frame_bgr: np.ndarray) -> FaceResult:
        url = self.base_url + self.identify_path
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"image": ("frame.jpg", self._jpg(frame_bgr), "image/jpeg")}
        r = requests.post(url, headers=headers, files=files, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        matches = data.get("matches") or data.get("results") or []
        if matches:
            best = max(matches, key=lambda m: m.get("confidence",0.0))
            ident = best.get("person") or best.get("name") or best.get("id")
            return FaceResult(ident, best.get("confidence"), data)
        return FaceResult(None, None, data)