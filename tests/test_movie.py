"""The movie renderer: every frame is written, as a GIF and (with ffmpeg) as an MP4."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sms_analysis import load_sms_csv, track_feature
from sms_analysis.movie import ffmpeg_path, frame_rows, render_movie, render_phase_movie, video_formats

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run():
    data = load_sms_csv(ROOT / "data" / "data.csv")
    return data, track_feature(data, start_wavelength=1542.3, kind="dip")


def test_gif_has_every_frame_and_changes_over_time(run, tmp_path):
    data, track = run
    out = render_movie(data, tmp_path / "m.gif", track=track, track_label="dip@1542.3", every=60, fps=10, dpi=50)
    rows = frame_rows(len(data), 60)
    with Image.open(out) as im:
        assert im.n_frames == len(rows)
        first = np.asarray(im.convert("RGB")).astype(int)
        im.seek(im.n_frames - 1)
        last = np.asarray(im.convert("RGB")).astype(int)
    # blitting must really redraw the moving artists: the last frame differs from the first, but
    # most of the picture (axes, map, legend = the background drawn once) is the same
    changed = (np.abs(last - first).sum(axis=2) > 30).mean()
    assert 0.002 < changed < 0.35


def test_follow_camera_falls_back_to_full_redraws(run, tmp_path):
    data, track = run
    out = render_movie(data, tmp_path / "f.gif", track=track, every=150, fps=10, dpi=40, follow_nm=6.0)
    with Image.open(out) as im:
        assert im.n_frames == len(frame_rows(len(data), 150))


def test_progress_is_reported_for_every_frame(run, tmp_path):
    data, track = run
    seen = []
    render_movie(data, tmp_path / "p.gif", track=track, every=150, dpi=40, progress=lambda i, n: seen.append((i, n)))
    n = len(frame_rows(len(data), 150))
    assert seen == [(i + 1, n) for i in range(n)]


def test_phase_movie_gif(run, tmp_path):
    data, _ = run
    out = render_phase_movie(data, tmp_path / "ph.gif", every=150, fps=10, dpi=40)
    with Image.open(out) as im:
        assert im.n_frames >= 4


def test_mp4_when_ffmpeg_is_available(run, tmp_path):
    data, track = run
    if ffmpeg_path() is None:
        assert video_formats() == ["gif"]
        out = render_movie(data, tmp_path / "m.mp4", track=track, every=150, dpi=40)
        assert out.suffix == ".gif"                    # no ffmpeg: a GIF is written instead
        return
    assert video_formats() == ["mp4", "gif"]
    out = render_movie(data, tmp_path / "m.mp4", track=track, every=100, fps=10, dpi=50)
    head = out.read_bytes()[:12]
    assert out.suffix == ".mp4" and head[4:8] == b"ftyp" and out.stat().st_size > 5000
