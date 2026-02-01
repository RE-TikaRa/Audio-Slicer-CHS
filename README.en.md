# Audio Slicer
A simple GUI application that slices audio with silence detection.

[Source project](https://github.com/flutydeer/audio-slicer)

[中文文档](./README.md)

## Screenshot

![image.png](https://s2.loli.net/2026/02/01/ONARIiVYwbsXgjF.png)

## Features

- Automatic slicing based on silence detection
- Preview of slice ranges and length distribution
- Output formats: wav / flac / mp3
- Multilingual UI
- Drag & drop audio import
- Dynamic threshold and VAD (voice activity detection)
- Parallel slicing (multi-thread / multi-process)
- Decode fallback: FFmpeg / Librosa
- Presets, naming rules, and slice list export (CSV / JSON)

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
- Presets: save/delete commonly used settings for quick reuse.
- Naming rules: optional prefix/suffix/timestamp for outputs.
- Export list: output CSV/JSON for slice ranges and paths.

## Parameters

- Threshold: areas below this RMS value are treated as silence, default -40 dB.
- Minimum Length: minimum slice length (ms), default 5000.
- Minimum Interval: minimum silence length for slicing (ms), default 300.
- Hop Size: RMS frame size (ms), default 10.
- Maximum Silence Length: max kept silence around slices (ms), default 1000.
- Dynamic Threshold: estimate noise floor from RMS distribution and apply an offset (dB).
- VAD: compensate for low-energy speech to avoid over-splitting.
- VAD Sensitivity: higher values keep quieter speech more easily.
- VAD Hangover: extra keep time after speech ends (ms).
- Parallel Mode / Jobs: choose serial / multi-thread / multi-process and worker count.
- Decode Fallback: strategy on read errors (ask / auto / skip).

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

Logs are written to the `log/` directory for diagnosis.

## License

See LICENSE in this repository.
The source project is MIT licensed.

## Localization

Localization by Re-TikaRa
