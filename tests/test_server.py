import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hrrmony.pipeline import CoverResult  # noqa: E402
from hrrmony.server.app import JobRunner, create_app  # noqa: E402


def fake_cover(song, out_dir, opts, progress):
    """Stands in for the GPU pipeline: walks the stages and writes small files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stage, frac in [("analyze", .05), ("separate", .2), ("convert", .6), ("master", .9)]:
        progress(stage, frac, stage)
    files = {k: out / f"x.{k}" for k in ("mp3", "wav")}
    files["original"] = out / "x_original.mp3"
    for p in files.values():
        p.write_bytes(b"ID3fake")
    return CoverResult(outputs=files, start=12.0, duration=opts.duration, timings={"total": 1.0})


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(JobRunner(tmp_path, run=fake_cover)))


def wait(client, job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health(client):
    h = client.get("/api/health").json()
    assert h["presets"] == {"classic": -8, "in-key": -12}


def test_upload_convert_download(client):
    r = client.post("/api/jobs", files={"file": ("song.mp3", b"\x00" * 1024, "audio/mpeg")},
                    data={"mode": "hook", "shift": "in-key"})
    assert r.status_code == 200, r.text
    assert r.json()["shift"] == -12
    job = wait(client, r.json()["id"])
    assert job["status"] == "done"
    assert job["result"]["start"] == 12.0
    assert set(job["files"]) == {"cover", "cover_wav", "original"}
    dl = client.get(job["files"]["cover"])
    assert dl.status_code == 200
    assert "villager" in dl.headers["content-disposition"]


@pytest.mark.parametrize("data,status", [
    ({"mode": "karaoke"}, 400),
    ({"shift": "loud"}, 400),
    ({"duration": "2"}, 400),
])
def test_bad_parameters(client, data, status):
    r = client.post("/api/jobs", files={"file": ("song.mp3", b"\x00", "audio/mpeg")}, data=data)
    assert r.status_code == status


def test_rejects_non_audio(client):
    r = client.post("/api/jobs", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Hrrmony" in r.text
