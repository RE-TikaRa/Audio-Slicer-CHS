# 音频切片机
一个简约的 GUI 应用程序，通过静音检测对音频进行切片。

[English README](./README.en.md)

## 截图

![image](./assets/screenshot_1.jpg)

## 功能

- 基于静音检测的自动切片
- 预览切片区间与长度分布图
- 输出格式：wav / flac / mp3
- 多语言界面
- 支持拖拽导入音频

## 快速开始

### Windows

- 发行版：在 GitHub Release 中下载并解压后运行 `slicer-gui.exe`。
- 源码运行：双击根目录的 `main.bat`。

### macOS & Linux

```shell
uv sync
uv run python scripts/slicer-gui.py
```
### 命令行

```shell
uv run python scripts/slicer.py path/to/audio.wav
```

## 使用说明

- 通过“Add Audio Files...”按钮或拖拽添加音频文件。
- 参数设置在右侧 Settings 面板内。
- 语言切换：主界面右侧 Settings 的 Language 下拉框。
- 勾选“Open output directory when finished”可在完成后自动打开输出目录。

## 参数说明

- Threshold（阈值）：RMS 低于此值的区域视为静音，默认 -40 dB。
- Minimum Length（最小长度）：每个切片的最小长度（ms），默认 5000。
- Minimum Interval（最小间隔）：静音段最小长度（ms），默认 300。
- Hop Size（步长）：RMS 帧长度（ms），默认 10。
- Maximum Silence Length（最大静音长度）：切片两端保留的最大静音长度（ms），默认 1000。

## 项目结构

- `src/audio_slicer/`：核心代码
  - `gui/`：主窗口与 UI
  - `utils/`：切片与预览工具
  - `modules/`：多语言文本
- `scripts/`：运行入口脚本
- `tools/`：打包与版本信息
- `assets/`：截图与 UI 文件
## 打包（Windows）

```pwsh
pwsh tools/pack-gui.ps1
```

输出目录：`dist/slicer-gui`，压缩包：`dist/slicer-gui-windows.zip`。

## 故障排查

- 出现 `Illegal Audio-MPEG-Header` 等 mpg123 错误：
  - 多见于损坏/非标准音频或扩展名与实际编码不一致。
  - 建议用 ffmpeg 转为 WAV/FLAC 后再切片。

- 预览失败或切片失败：
  - 请确认文件可被正常解码（能被其他播放器正常播放）。

日志会写入根目录 `log/` 目录，便于排查问题。

## 许可

本项目遵循仓库中的 LICENSE。

## 汉化

汉化 by Re-TikaRa
