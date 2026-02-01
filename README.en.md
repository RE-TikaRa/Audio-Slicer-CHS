# Audio Slicer
A simple GUI application that slices audio with silence detection.

[Source project](https://github.com/flutydeer/audio-slicer)

[中文文档](./README.md)

## Screenshot

![image.png](https://s2.loli.net/2026/02/01/ONARIiVYwbsXgjF.png)

## Features

- Automatic slicing based on silence detection
- Preview of slice ranges and length distribution (separate window + zoom)
- Output formats: wav / flac / mp3
- Multilingual UI
- Drag & drop audio import
- Dynamic threshold and VAD (voice activity detection)
- Parallel slicing (multi-thread / multi-process)
- Decode fallback: FFmpeg / Librosa
- Presets, naming rules, and slice list export (CSV / JSON)
- Settings tabs (Basic / Advanced)

## Quick Start

### Windows

- Release: download from GitHub Releases and run `slicer-gui.exe`.
- Source: double-click `main.bat` in the project root.

### macOS & Linux

```shell
uv sync
uv run python scripts/slicer-gui.py
```
### CLI

```shell
uv run python scripts/slicer.py path/to/audio.wav
```

## Usage

- Add audio files by clicking “Add Audio Files...” or drag & drop them into the window.
- Parameters are on the Settings panel at the right.
- Language switch: use the Language dropdown in the Settings panel.
- Enable “Open output directory when finished” to open the output folder automatically.
- Presets: save/delete/restore defaults, with a completion prompt after reset.
- Naming rules: optional prefix/suffix/timestamp for outputs.
- Export list: output CSV/JSON for slice ranges and paths.
- The Preview button opens a separate window with a zoom slider and mouse-wheel zoom.
- Settings are split into “Basic / Advanced”; advanced options include parallelism, fallback, dynamic threshold, and VAD.

## Parameters

- Threshold: areas below this RMS value are treated as silence, default -40 dB.
- Minimum Length: minimum slice length (ms), default 5000.
- Minimum Interval: minimum silence length for slicing (ms), default 300.
- Hop Size: RMS frame size (ms), default 10.
- Maximum Silence Length: max kept silence around slices (ms), default 1000.
- Dynamic Threshold: estimate noise floor from RMS distribution and apply an offset (dB).
- Dynamic Offset: offset for dynamic threshold (dB), higher is stricter.
- VAD: compensate for low-energy speech to avoid over-splitting.
- VAD Sensitivity: higher values keep quieter speech more easily.
- VAD Hangover: extra keep time after speech ends (ms).
- Parallel Mode / Jobs: choose serial / multi-thread / multi-process and worker count.
- Decode Fallback: strategy on read errors (ask / auto / skip).

## FFmpeg Notes

Fallback decoder lookup order:

1. Env var `AUDIO_SLICER_FFMPEG` points to `ffmpeg.exe`
2. Repo root: `ffmpeg.exe`
3. Repo root: `ffmpeg/bin/ffmpeg.exe`
4. `tools/ffmpeg.exe`
5. `tools/ffmpeg/ffmpeg.exe`
6. `tools/ffmpeg/bin/ffmpeg.exe`
7. `ffmpeg` in system PATH

## Project Structure

- `src/audio_slicer/`: core code
  - `gui/`: main window and UI
  - `utils/`: slicing and preview utilities
  - `modules/`: i18n strings
- `scripts/`: entry scripts
- `tools/`: packaging and version info
- `assets/`: screenshots and UI files
## Packaging (Windows)

```pwsh
pwsh tools/pack-gui.ps1
```

Output: `dist/slicer-gui`, ZIP: `dist/slicer-gui-windows.zip`.

## Troubleshooting

- `Illegal Audio-MPEG-Header` or mpg123 resync errors:
  - Often caused by corrupted/non-standard audio or mismatched extension/codec.
  - Convert to WAV/FLAC with ffmpeg and try again.

- Preview or slicing fails:
  - Ensure the file can be decoded (playable in other players).

- Chinese text rendered as squares in preview:
  - Ensure `Microsoft YaHei` or `SimHei` is installed.
  - Re-generate the preview.

Logs are written to the `log/` directory for diagnosis.

## License

See LICENSE in this repository.
The source project is MIT licensed.

## Localization

Localization by Re-TikaRa
