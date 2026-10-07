"""Local web app: upload a song, get a villager cover back."""

import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import SHIFT_PRESETS, home

STATIC = Path(__file__).parent / "static"
ALLOWED = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mp4"}
MAX_BYTES = 300 * 1024 * 1024


@dataclass
class Job:
    id: str
    filename: str
    mode: str
    shift: int
    duration: float
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting for the GPU"
    created: float = field(default_factory=time.time)
    result: dict[str, Any] | None = None
    error: str | None = None
    files: dict[str, Path] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "filename": self.filename, "mode": self.mode, "shift": self.shift,
            "status": self.status, "stage": self.stage, "progress": round(self.progress, 3),
            "message": self.message, "error": self.error, "result": self.result,
            "files": {k: f"/api/jobs/{self.id}/files/{k}" for k in self.files},
        }


class JobRunner:
    """One worker thread: covers run one at a time (they share the GPU and model caches)."""

    def __init__(self, root: Path, run=None):
        from ..pipeline import make_cover

        self.root = root
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue[tuple[Job, Path]] = queue.Queue()
        self._run = run or make_cover
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, job: Job, upload: Path) -> Job:
        self.jobs[job.id] = job
        self.q.put((job, upload))
        return job

    def _loop(self) -> None:
        from ..pipeline import CoverOptions

        while True:
            job, upload = self.q.get()
            job.status, job.message = "running", "Starting"

            def progress(stage: str, frac: float, msg: str, job=job) -> None:
                job.stage, job.progress, job.message = stage, frac, msg

            try:
                opts = CoverOptions(mode=job.mode, duration=job.duration, shift=job.shift,
                                    formats=("mp3", "wav"))
                res = self._run(upload, self.root / job.id, opts, progress)
                job.files = {"cover": res.outputs["mp3"], "cover_wav": res.outputs["wav"]}
                if "original" in res.outputs:
                    job.files["original"] = res.outputs["original"]
                job.result = {"start": res.start, "duration": res.duration,
                              "seconds": round(res.timings.get("total", 0.0), 1)}
                job.status, job.stage, job.progress = "done", "done", 1.0
            except Exception as exc:  # surfaced to the UI
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                job.message = "Something went wrong"
                traceback.print_exc()


def create_app(runner: JobRunner | None = None) -> FastAPI:
    # (no `from __future__ import annotations` here: FastAPI resolves the route signatures)
    root = home() / "web"
    root.mkdir(parents=True, exist_ok=True)
    runner = runner or JobRunner(root)
    app = FastAPI(title="villager-sound", version=__version__)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            import torch

            device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        except Exception:
            device = "unknown"
        return {"version": __version__, "device": device, "presets": SHIFT_PRESETS}

    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...), mode: str = Form("hook"),
                         shift: str = Form("classic"), duration: float = Form(30.0)):
        if mode not in {"hook", "full"}:
            raise HTTPException(400, "mode must be 'hook' or 'full'")
        semis = SHIFT_PRESETS.get(shift)
        if semis is None:
            try:
                semis = int(shift)
            except ValueError:
                raise HTTPException(400, "shift must be 'classic', 'in-key' or semitones") from None
        if not 5 <= duration <= 120:
            raise HTTPException(400, "duration must be between 5 and 120 seconds")
        name = Path(file.filename or "song").name
        if Path(name).suffix.lower() not in ALLOWED:
            raise HTTPException(415, f"unsupported file type; use one of {sorted(ALLOWED)}")
        job = Job(id=uuid.uuid4().hex[:12], filename=name, mode=mode, shift=semis,
                  duration=duration)
        dest = root / job.id / f"input{Path(name).suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with dest.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    fh.close()
                    shutil.rmtree(dest.parent, ignore_errors=True)
                    raise HTTPException(413, "file too large (max 300 MB)")
                fh.write(chunk)
        runner.submit(job, dest)
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = runner.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.get("/api/jobs/{job_id}/files/{kind}")
    def get_file(job_id: str, kind: str):
        job = runner.jobs.get(job_id)
        if not job or kind not in job.files:
            raise HTTPException(404, "not found")
        path = job.files[kind]
        stem = Path(job.filename).stem
        nice = f"{stem} (villager).{path.suffix.lstrip('.')}" if kind.startswith("cover") \
            else f"{stem} (original excerpt){path.suffix}"
        return FileResponse(path, filename=nice)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
