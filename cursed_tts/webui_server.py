"""
Local WebUI server for the two-swarm lab.

    python -m cursed_tts webui

Serves webui/ and POST /api/speak with a full picker/speaker trace.
Does not retrain. Freeze inference only.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .trace import load_freeze_models, trace_text

_ROOT = Path(__file__).resolve().parent.parent
_WEBUI = _ROOT / "webui"
_ASSETS = _ROOT / "assets"


_picker = None
_speakers = None
_lock = threading.Lock()


def _ensure_models(picker_path: str, speaker_path: str):
    global _picker, _speakers
    with _lock:
        if _picker is None or _speakers is None:
            _picker, _speakers = load_freeze_models(picker_path, speaker_path)
    return _picker, _speakers


def _safe_file(root: Path, rel: str) -> Path | None:
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target if target.is_file() else None


class Handler(BaseHTTPRequestHandler):
    picker_path = "model_more_fly_best.npz"
    speaker_path = "model_speaker.npz"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[webui] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            target = _WEBUI / "index.html"
        elif path.startswith("/assets/"):
            target = _safe_file(_ASSETS, path[len("/assets/") :])
            if target is None:
                self._send(404, b"not found", "text/plain")
                return
        else:
            target = _safe_file(_WEBUI, path.lstrip("/"))
            if target is None:
                self._send(404, b"not found", "text/plain")
                return
        if not Path(target).is_file():
            self._send(404, b"not found", "text/plain")
            return
        data = Path(target).read_bytes()
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, data, ctype)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/speak":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
            text = str(payload.get("text") or "").strip()
            if not text:
                raise ValueError("text is required")
            if len(text) > 2000:
                raise ValueError("text too long (max 2000 chars)")
            picker, speakers = _ensure_models(self.picker_path, self.speaker_path)
            trace = trace_text(picker, speakers, text)
            body = json.dumps(trace).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self._send(400, err, "application/json")


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    picker_path: str = "model_more_fly_best.npz",
    speaker_path: str = "model_speaker.npz",
) -> None:
    if not _WEBUI.is_dir():
        raise SystemExit(f"Missing webui directory at {_WEBUI}")
    Handler.picker_path = picker_path
    Handler.speaker_path = speaker_path
    print(f"Loading freeze models ({picker_path}, {speaker_path})…")
    _ensure_models(picker_path, speaker_path)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Flysune lab → http://{host}:{port}")
    print('  POST /api/speak  {"text": "..."}')
    print("  Speakers never see KC. Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        httpd.server_close()
