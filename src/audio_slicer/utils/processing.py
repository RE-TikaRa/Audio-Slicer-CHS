import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile

from audio_slicer.modules import i18n
from audio_slicer.utils.slicer2 import Slicer, estimate_dynamic_threshold_db, build_vad_mask


def resolve_ffmpeg_path() -> str | None:
    env_path = os.environ.get("AUDIO_SLICER_FFMPEG")
    if env_path and os.path.isfile(env_path):
        return env_path
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parents[3]
    candidates = [
        base_dir / "ffmpeg.exe",
        base_dir / "ffmpeg" / "bin" / "ffmpeg.exe",
        base_dir / "tools" / "ffmpeg.exe",
        base_dir / "tools" / "ffmpeg" / "ffmpeg.exe",
        base_dir / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


def _read_with_ffmpeg(filename: str, ffmpeg_path: str) -> tuple[np.ndarray | None, int | None, str | None]:
    with tempfile.NamedTemporaryFile(
        prefix="audio_slicer_decode_",
        suffix=".wav",
        delete=False,
    ) as tmp:
        temp_path = tmp.name
    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                filename,
                "-vn",
                "-acodec",
                "pcm_s16le",
                temp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None, None, result.stderr.strip() or result.stdout.strip()
        audio, sr = soundfile.read(temp_path, dtype=np.float32)
        return audio, sr, None
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _read_with_librosa(filename: str) -> tuple[np.ndarray | None, int | None, str | None]:
    try:
        import librosa
    except Exception as exc:
        return None, None, str(exc)
    try:
        audio, sr = librosa.load(filename, sr=None, mono=False)
        return audio, sr, None
    except Exception as exc:
        return None, None, str(exc)


def _prepare_audio(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    is_mono = True
    if audio.ndim > 1:
        is_mono = False
        audio = audio.T
    return audio, is_mono


def process_audio_file(
    filename: str,
    *,
    output_ext: str,
    threshold_db: float,
    min_length: int,
    min_interval: int,
    hop_size: int,
    max_silence: int,
    dynamic_enabled: bool,
    dynamic_offset_db: float,
    vad_enabled: bool,
    vad_sensitivity_db: float,
    vad_hangover_ms: int,
    name_prefix: str,
    name_suffix: str,
    name_timestamp: bool,
    export_csv: bool,
    export_json: bool,
    output_dir: str | None,
    fallback_mode: str,
    language: str,
) -> tuple[bool, str | None, str | None]:
    error = None
    try:
        audio, sr = soundfile.read(filename, dtype=np.float32)
    except Exception as exc:
        error = str(exc)
        if fallback_mode not in {"ffmpeg", "librosa", "ffmpeg_then_librosa"}:
            return False, error, None
        audio = None
        sr = None
    if audio is None:
        if fallback_mode in {"ffmpeg", "ffmpeg_then_librosa"}:
            ffmpeg_path = resolve_ffmpeg_path()
            if not ffmpeg_path:
                if fallback_mode == "ffmpeg":
                    return False, i18n.text("ffmpeg_not_found", language), None
            else:
                audio, sr, ffmpeg_error = _read_with_ffmpeg(filename, ffmpeg_path)
                if audio is None and fallback_mode == "ffmpeg":
                    return False, i18n.text("ffmpeg_failed", language).format(error=ffmpeg_error or ""), None
                if audio is None:
                    error = ffmpeg_error
        if audio is None and fallback_mode in {"librosa", "ffmpeg_then_librosa"}:
            audio, sr, librosa_error = _read_with_librosa(filename)
            if audio is None:
                return False, librosa_error or error or "Decode failed.", None
    if audio is None or sr is None:
        return False, error or "Decode failed.", None

    audio, is_mono = _prepare_audio(audio)
    slicer = Slicer(
        sr=sr,
        threshold=threshold_db,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_silence,
    )
    rms_list = None
    dynamic_threshold_db = None
    vad_mask = None
    if dynamic_enabled or vad_enabled:
        rms_list = slicer.get_rms_list(audio)
    if dynamic_enabled and rms_list is not None:
        dynamic_threshold_db = estimate_dynamic_threshold_db(rms_list, offset_db=dynamic_offset_db)
    if vad_enabled and rms_list is not None:
        base_threshold = dynamic_threshold_db if dynamic_threshold_db is not None else slicer.threshold_db
        hangover_frames = 0
        if vad_hangover_ms > 0 and hop_size > 0:
            hangover_frames = max(1, int(round(vad_hangover_ms / hop_size)))
        vad_mask = build_vad_mask(
            rms_list,
            threshold_db=base_threshold,
            sensitivity_db=vad_sensitivity_db,
            hangover_frames=hangover_frames,
        )
    sil_tags, total_frames, _ = slicer.get_slice_tags(
        audio,
        dynamic_threshold_db=dynamic_threshold_db,
        vad_mask=vad_mask,
        rms_list=rms_list,
    )
    chunks = slicer.slice(audio, sil_tags, total_frames)

    hop_ms = hop_size
    ranges = _get_ranges(sil_tags, total_frames, hop_ms)
    out_dir = output_dir or os.path.dirname(os.path.abspath(filename))
    info = Path(out_dir)
    info.mkdir(parents=True, exist_ok=True)

    base_name = os.path.basename(filename).rsplit(".", maxsplit=1)[0]
    file_core = f"{name_prefix}{base_name}{name_suffix}"
    if name_timestamp:
        time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_core = f"{file_core}_{time_tag}"

    slice_records = []
    for i, chunk in enumerate(chunks):
        path = os.path.join(out_dir, f"{file_core}_{i}.{output_ext}")
        if not is_mono:
            chunk = chunk.T
        soundfile.write(path, chunk, sr)
        if i < len(ranges):
            start_ms, end_ms = ranges[i]
        else:
            start_ms, end_ms = None, None
        length_ms = (end_ms - start_ms) if start_ms is not None and end_ms is not None else None
        slice_records.append(
            {
                "index": i,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "length_ms": length_ms,
                "output_path": path,
                "source_file": filename,
            }
        )

    if export_csv:
        csv_path = os.path.join(out_dir, f"{file_core}_slices.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "start_ms", "end_ms", "length_ms", "output_path", "source_file"],
            )
            writer.writeheader()
            for record in slice_records:
                writer.writerow({key: "" if value is None else value for key, value in record.items()})
    if export_json:
        json_path = os.path.join(out_dir, f"{file_core}_slices.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(slice_records, f, ensure_ascii=False, indent=2)

    return True, None, str(out_dir)


def _get_ranges(sil_tags, total_frames: int, hop_ms: int):
    if len(sil_tags) == 0:
        return [(0, total_frames * hop_ms)]
    ranges = []
    if sil_tags[0][0] > 0:
        ranges.append((0, sil_tags[0][0] * hop_ms))
    for i in range(len(sil_tags) - 1):
        ranges.append((sil_tags[i][1] * hop_ms, sil_tags[i + 1][0] * hop_ms))
    if sil_tags[-1][1] < total_frames:
        ranges.append((sil_tags[-1][1] * hop_ms, total_frames * hop_ms))
    return ranges
