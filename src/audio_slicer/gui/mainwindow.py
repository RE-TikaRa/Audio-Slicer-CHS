import json
import os
import subprocess
import tempfile

import soundfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from typing import List
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from audio_slicer.utils.slicer2 import Slicer, estimate_dynamic_threshold_db, build_vad_mask
from audio_slicer.utils.processing import process_audio_file, resolve_ffmpeg_path

from audio_slicer.gui.Ui_MainWindow import Ui_MainWindow
from audio_slicer.utils.preview import SlicingPreview
from audio_slicer.modules import i18n

APP_VERSION = "1.4.0"


class _FallbackBridge(QObject):
    request = Signal(str, str)

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        self.choice: str | None = None
        self.request.connect(self._on_request)

    @Slot(str, str)
    def _on_request(self, filename: str, error: str):
        choice = self.window._show_fallback_dialog("process_read_failed", filename, error)
        self.mutex.lock()
        self.choice = choice
        self.cond.wakeAll()
        self.mutex.unlock()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btnAddFiles.clicked.connect(self._on_add_audio_files)
        self.ui.btnBrowse.clicked.connect(self._on_browse_output_dir)
        self.ui.btnRemove.clicked.connect(self._on_remove_audio_file)
        self.ui.btnClearList.clicked.connect(self._on_clear_audio_list)
        self.ui.btnAbout.clicked.connect(self._on_about)
        self.ui.btnPreviewSelection.clicked.connect(self._on_preview_selection)
        self.ui.btnStart.clicked.connect(self._on_start)
        # self.ui.twImages.tabCloseRequested.connect(self._on_tab_close_requested)

        # tab = QLabel()
        # pix = QPixmap('preview.png')
        # scalH = int(tab.height() * 1.25)
        # scalW = pix.scaledToHeight(scalH).width()
        # if (self.ui.twImages.width() < scalW):
        #     scalW = int(tab.width() * 1.25)
        #     scalH = pix.scaledToWidth(scalW).height()
        # tab.setPixmap(pix.scaled(scalW, scalH))
        # self.ui.twImages.addTab(tab, "preview.png")

        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(100)
        self.ui.progressBar.setValue(0)
        self.ui.btnStart.setDefault(True)

        validator = QRegularExpressionValidator(QRegularExpression(r"\d+"))
        self.ui.leThreshold.setValidator(QDoubleValidator())
        self.ui.leMinLen.setValidator(validator)
        self.ui.leMinInterval.setValidator(validator)
        self.ui.leHopSize.setValidator(validator)
        self.ui.leMaxSilence.setValidator(validator)

        self.ui.lwTaskList.setAlternatingRowColors(True)

        # State variables
        self.workers: list[QThread] = []
        self.workCount = 0
        self.workFinished = 0
        self.processing = False
        self.last_output_dir: str | None = None

        # Language setup
        self.current_language = i18n.normalize_language(QLocale.system().name())
        self._init_language_selector()
        self._init_extra_ui()
        self._load_presets()
        self._apply_language()
        self._fallback_bridge = _FallbackBridge(self)
        self._fallback_request_lock = QMutex()

        # Must set to accept drag and drop events
        self.setAcceptDrops(True)

        # Get available formats/extensions supported
        self.availableFormats = [str(formatExt).lower()
                                 for formatExt in soundfile.available_formats().keys()]
        # libsndfile supports Opus in Ogg container
        # .opus is a valid extension and recommended for Ogg Opus (see RFC 7845, Section 9)
        # append opus for convenience as tools like youtube-dl(p) extract to .opus by default
        self.availableFormats.append("opus")
        self.formatAllFilter = " ".join(
            [f"*.{formatExt}" for formatExt in self.availableFormats])
        self.formatIndividualFilter = ";;".join(
            [f"{formatExt} (*.{formatExt})" for formatExt in sorted(self.availableFormats)])

    def _on_tab_close_requested(self, index):
        self.ui.twImages.removeTab(index)

    def _on_browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Browse Output Directory", ".")
        if path != "":
            self.ui.leOutputDir.setText(QDir.toNativeSeparators(path))

    def _on_add_audio_files(self):
        if self.processing:
            self._warningProcessNotFinished()
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            i18n.text("select_audio_files", self.current_language),
            ".",
            f"Audio ({self.formatAllFilter});;{self.formatIndividualFilter}",
        )
        for path in paths:
            item = QListWidgetItem()
            item.setSizeHint(QSize(200, 24))
            item.setText(QFileInfo(path).fileName())
            # Save full path at custom role
            item.setData(Qt.ItemDataRole.UserRole + 1, path)
            self.ui.lwTaskList.addItem(item)

    def _on_remove_audio_file(self):
        item = self.ui.lwTaskList.currentItem()
        if item is None:
            return
        self.ui.lwTaskList.takeItem(self.ui.lwTaskList.row(item))
        return

    def _on_clear_audio_list(self):
        if self.processing:
            self._warningProcessNotFinished()
            return

        self.ui.lwTaskList.clear()

    def _on_about(self):
        language_label = i18n.LANGUAGES.get(self.current_language, self.current_language)
        QMessageBox.information(
            self,
            i18n.text("about", self.current_language),
            i18n.text("about_text", self.current_language).format(
                version=APP_VERSION,
                language=language_label,
            ),
        )

    def _on_start(self):
        if self.processing:
            self._warningProcessNotFinished()
            return

        item_count = self.ui.lwTaskList.count()
        if item_count == 0:
            return

        output_format = self._get_output_format()
        if output_format == "mp3":
            ret = QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("mp3_warning", self.current_language),
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if ret == QMessageBox.Cancel:
                return

        options = self._collect_processing_options()
        if options["parallel_mode"] == "process" and options["fallback_mode"] == "ask":
            ret = QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("parallel_process_fallback_hint", self.current_language),
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if ret == QMessageBox.Cancel:
                return
            options["fallback_mode"] = "ffmpeg_then_librosa"

        class WorkThread(QThread):
            oneFinished = Signal()
            errorOccurred = Signal(str, str)

            def __init__(self, filenames: List[str], window: MainWindow, output_ext: str, options: dict):
                super().__init__()

                self.filenames = filenames
                self.win = window
                self.output_ext = output_ext
                self.options = options

            def run(self):
                mode = self.options["parallel_mode"]
                if mode == "single":
                    for filename in self.filenames:
                        try:
                            self._process_file(filename)
                        finally:
                            self.oneFinished.emit()
                    return
                if mode == "thread":
                    with ThreadPoolExecutor(max_workers=self.options["parallel_jobs"]) as executor:
                        futures = {
                            executor.submit(self._process_file, filename): filename
                            for filename in self.filenames
                        }
                        for future in as_completed(futures):
                            filename = futures[future]
                            try:
                                future.result()
                            except Exception as exc:
                                self.errorOccurred.emit(filename, str(exc))
                            finally:
                                self.oneFinished.emit()
                    return
                with ProcessPoolExecutor(max_workers=self.options["parallel_jobs"]) as executor:
                    futures = {
                        executor.submit(
                            process_audio_file,
                            filename,
                            **self._build_process_kwargs(),
                        ): filename
                        for filename in self.filenames
                    }
                    for future in as_completed(futures):
                        filename = futures[future]
                        try:
                            ok, error, out_dir = future.result()
                            if ok:
                                if out_dir:
                                    self.win.last_output_dir = out_dir
                            else:
                                self.errorOccurred.emit(filename, error or "Unknown error.")
                        except Exception as exc:
                            self.errorOccurred.emit(filename, str(exc))
                        finally:
                            self.oneFinished.emit()

            def _process_file(self, filename: str) -> bool:
                if self.options["fallback_mode"] == "ask":
                    try:
                        ok, error, out_dir = process_audio_file(
                            filename,
                            **self._build_process_kwargs(fallback_mode="skip"),
                        )
                    except Exception as exc:
                        self.errorOccurred.emit(filename, str(exc))
                        return False
                    if ok:
                        if out_dir:
                            self.win.last_output_dir = out_dir
                        return True
                    choice = self.win._request_fallback_choice(filename, error or "")
                    if choice == "ffmpeg":
                        try:
                            ok, error, out_dir = process_audio_file(
                                filename,
                                **self._build_process_kwargs(fallback_mode="ffmpeg"),
                            )
                        except Exception as exc:
                            self.errorOccurred.emit(filename, str(exc))
                            return False
                    elif choice == "librosa":
                        try:
                            ok, error, out_dir = process_audio_file(
                                filename,
                                **self._build_process_kwargs(fallback_mode="librosa"),
                            )
                        except Exception as exc:
                            self.errorOccurred.emit(filename, str(exc))
                            return False
                    else:
                        self.errorOccurred.emit(
                            filename,
                            i18n.text("skipped_by_user", self.win.current_language),
                        )
                        return False
                    if ok:
                        if out_dir:
                            self.win.last_output_dir = out_dir
                        return True
                    self.errorOccurred.emit(filename, error or "Unknown error.")
                    return False

                try:
                    ok, error, out_dir = process_audio_file(
                        filename,
                        **self._build_process_kwargs(),
                    )
                except Exception as exc:
                    self.errorOccurred.emit(filename, str(exc))
                    return False
                if ok:
                    if out_dir:
                        self.win.last_output_dir = out_dir
                    return True
                self.errorOccurred.emit(filename, error or "Unknown error.")
                return False

            def _build_process_kwargs(self, fallback_mode: str | None = None) -> dict:
                opts = self.options
                return {
                    "output_ext": self.output_ext,
                    "threshold_db": opts["threshold_db"],
                    "min_length": opts["min_length"],
                    "min_interval": opts["min_interval"],
                    "hop_size": opts["hop_size"],
                    "max_silence": opts["max_silence"],
                    "dynamic_enabled": opts["dynamic_enabled"],
                    "dynamic_offset_db": opts["dynamic_offset_db"],
                    "vad_enabled": opts["vad_enabled"],
                    "vad_sensitivity_db": opts["vad_sensitivity_db"],
                    "vad_hangover_ms": opts["vad_hangover_ms"],
                    "name_prefix": opts["name_prefix"],
                    "name_suffix": opts["name_suffix"],
                    "name_timestamp": opts["name_timestamp"],
                    "export_csv": opts["export_csv"],
                    "export_json": opts["export_json"],
                    "output_dir": opts["output_dir"],
                    "fallback_mode": fallback_mode or opts["fallback_mode"],
                    "language": self.win.current_language,
                }

        # Collect paths
        paths: list[str] = []
        for i in range(0, item_count):
            item = self.ui.lwTaskList.item(i)
            path = item.data(Qt.ItemDataRole.UserRole + 1)  # Get full path
            paths.append(path)

        self.ui.progressBar.setMaximum(item_count)
        self.ui.progressBar.setValue(0)

        self.workCount = item_count
        self.workFinished = 0
        self._setProcessing(True)

        # Start work thread
        worker = WorkThread(paths, self, output_format, options)
        worker.oneFinished.connect(self._oneFinished)
        worker.errorOccurred.connect(self._on_worker_error)
        worker.finished.connect(self._threadFinished)
        worker.start()

        self.workers.append(worker)  # Collect in case of auto deletion

    def _oneFinished(self):
        self.workFinished += 1
        self.ui.progressBar.setValue(self.workFinished)

    def _on_worker_error(self, filename: str, error: str):
        QMessageBox.warning(
            self,
            i18n.text("warning_title", self.current_language),
            i18n.text("read_failed", self.current_language).format(
                file=filename,
                error=error,
            ),
        )

    def _request_fallback_choice(self, filename: str, error: str) -> str:
        self._fallback_request_lock.lock()
        try:
            bridge = self._fallback_bridge
            bridge.mutex.lock()
            bridge.choice = None
            bridge.request.emit(filename, error)
            while bridge.choice is None:
                bridge.cond.wait(bridge.mutex)
            choice = bridge.choice or "cancel"
            bridge.mutex.unlock()
            return choice
        finally:
            self._fallback_request_lock.unlock()

    def _threadFinished(self):
        # Join all workers
        for worker in self.workers:
            worker.wait()
        self.workers.clear()
        self._setProcessing(False)

        QMessageBox.information(
            self,
            QApplication.applicationName(),
            i18n.text("slicing_complete", self.current_language),
        )
        if self.ui.cbxOpenOutuptDirectory.isChecked() and self.last_output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output_dir))

    def _warningProcessNotFinished(self):
        QMessageBox.warning(
            self,
            QApplication.applicationName(),
            i18n.text("process_not_finished", self.current_language),
        )

    def _setProcessing(self, processing: bool):
        is_enabled = not processing
        self.ui.btnStart.setText(
            i18n.text("slicing", self.current_language) if processing else i18n.text("start", self.current_language))
        self.ui.btnStart.setEnabled(is_enabled)
        self.ui.btnPreviewSelection.setEnabled(is_enabled)
        self.ui.btnAddFiles.setEnabled(is_enabled)
        self.ui.lwTaskList.setEnabled(is_enabled)
        self.ui.btnClearList.setEnabled(is_enabled)
        self.ui.leThreshold.setEnabled(is_enabled)
        self.ui.leMinLen.setEnabled(is_enabled)
        self.ui.leMinInterval.setEnabled(is_enabled)
        self.ui.leHopSize.setEnabled(is_enabled)
        self.ui.leMaxSilence.setEnabled(is_enabled)
        self.ui.leOutputDir.setEnabled(is_enabled)
        self.ui.btnBrowse.setEnabled(is_enabled)
        self.ui.cbLanguage.setEnabled(is_enabled)
        self.ui.cbPresets.setEnabled(is_enabled)
        self.ui.btnPresetSave.setEnabled(is_enabled)
        self.ui.btnPresetDelete.setEnabled(is_enabled)
        self.ui.leNamePrefix.setEnabled(is_enabled)
        self.ui.leNameSuffix.setEnabled(is_enabled)
        self.ui.cbxNameTimestamp.setEnabled(is_enabled)
        self.ui.cbxExportCsv.setEnabled(is_enabled)
        self.ui.cbxExportJson.setEnabled(is_enabled)
        self.ui.cbxDynamicThreshold.setEnabled(is_enabled)
        self.ui.leDynamicOffset.setEnabled(is_enabled)
        self.ui.cbxVAD.setEnabled(is_enabled)
        self.ui.leVADSensitivity.setEnabled(is_enabled)
        self.ui.leVADHangover.setEnabled(is_enabled)
        self.ui.cbParallelMode.setEnabled(is_enabled)
        self.ui.sbParallelJobs.setEnabled(is_enabled)
        self.ui.cbFallbackMode.setEnabled(is_enabled)
        self.processing = processing

    def _init_extra_ui(self):
        self._init_preview_panel()
        self._init_settings_tabs()
        self._init_advanced_controls()

    def _init_preview_panel(self):
        self.groupBoxPreview = QGroupBox(self)
        self.groupBoxPreview.setObjectName("groupBoxPreview")
        preview_layout = QVBoxLayout(self.groupBoxPreview)
        self.labelPreview = QLabel(self.groupBoxPreview)
        self.labelPreview.setObjectName("labelPreview")
        self.labelPreview.setAlignment(Qt.AlignCenter)
        self.labelPreview.setWordWrap(True)
        self.labelPreview.setMinimumSize(240, 240)
        self.labelPreview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.labelPreview)
        self.ui.horizontalLayout.addWidget(self.groupBoxPreview)
        self._preview_pixmap_path: str | None = None

    def _init_settings_tabs(self):
        self.ui.settingsTabs = QTabWidget(self.ui.groupBox_2)
        self.ui.tabBasic = QWidget(self.ui.settingsTabs)
        self.ui.tabAdvanced = QWidget(self.ui.settingsTabs)
        self.ui.basicFormLayout = self.ui.formLayout
        self.ui.advancedFormLayout = QFormLayout()

        self.ui.verticalLayout_3.removeItem(self.ui.formLayout)
        self.ui.tabBasic.setLayout(self.ui.basicFormLayout)
        self.ui.tabAdvanced.setLayout(self.ui.advancedFormLayout)
        self.ui.settingsTabs.addTab(self.ui.tabBasic, "")
        self.ui.settingsTabs.addTab(self.ui.tabAdvanced, "")
        self.ui.verticalLayout_3.insertWidget(0, self.ui.settingsTabs)

    def _init_advanced_controls(self):
        advanced_layout = self.ui.advancedFormLayout
        self.ui.labelPreset = QLabel(self.ui.groupBox_2)
        self.ui.cbPresets = QComboBox(self.ui.groupBox_2)
        self.ui.btnPresetSave = QPushButton(self.ui.groupBox_2)
        self.ui.btnPresetDelete = QPushButton(self.ui.groupBox_2)
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(self.ui.cbPresets)
        preset_layout.addWidget(self.ui.btnPresetSave)
        preset_layout.addWidget(self.ui.btnPresetDelete)
        preset_widget = QWidget(self.ui.groupBox_2)
        preset_widget.setLayout(preset_layout)
        advanced_layout.addRow(self.ui.labelPreset, preset_widget)

        self.ui.labelNamePrefix = QLabel(self.ui.groupBox_2)
        self.ui.leNamePrefix = QLineEdit(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelNamePrefix, self.ui.leNamePrefix)

        self.ui.labelNameSuffix = QLabel(self.ui.groupBox_2)
        self.ui.leNameSuffix = QLineEdit(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelNameSuffix, self.ui.leNameSuffix)

        self.ui.labelNameTimestamp = QLabel(self.ui.groupBox_2)
        self.ui.cbxNameTimestamp = QCheckBox(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelNameTimestamp, self.ui.cbxNameTimestamp)

        self.ui.labelExportCsv = QLabel(self.ui.groupBox_2)
        self.ui.cbxExportCsv = QCheckBox(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelExportCsv, self.ui.cbxExportCsv)

        self.ui.labelExportJson = QLabel(self.ui.groupBox_2)
        self.ui.cbxExportJson = QCheckBox(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelExportJson, self.ui.cbxExportJson)

        self.ui.labelDynamicThreshold = QLabel(self.ui.groupBox_2)
        self.ui.cbxDynamicThreshold = QCheckBox(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelDynamicThreshold, self.ui.cbxDynamicThreshold)

        self.ui.labelDynamicOffset = QLabel(self.ui.groupBox_2)
        self.ui.leDynamicOffset = QLineEdit(self.ui.groupBox_2)
        self.ui.leDynamicOffset.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        advanced_layout.addRow(self.ui.labelDynamicOffset, self.ui.leDynamicOffset)

        self.ui.labelVAD = QLabel(self.ui.groupBox_2)
        self.ui.cbxVAD = QCheckBox(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelVAD, self.ui.cbxVAD)

        self.ui.labelVADSensitivity = QLabel(self.ui.groupBox_2)
        self.ui.leVADSensitivity = QLineEdit(self.ui.groupBox_2)
        self.ui.leVADSensitivity.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        advanced_layout.addRow(self.ui.labelVADSensitivity, self.ui.leVADSensitivity)

        self.ui.labelVADHangover = QLabel(self.ui.groupBox_2)
        self.ui.leVADHangover = QLineEdit(self.ui.groupBox_2)
        self.ui.leVADHangover.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        advanced_layout.addRow(self.ui.labelVADHangover, self.ui.leVADHangover)

        self.ui.labelParallelMode = QLabel(self.ui.groupBox_2)
        self.ui.cbParallelMode = QComboBox(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelParallelMode, self.ui.cbParallelMode)

        self.ui.labelParallelJobs = QLabel(self.ui.groupBox_2)
        self.ui.sbParallelJobs = QSpinBox(self.ui.groupBox_2)
        self.ui.sbParallelJobs.setRange(1, max(1, (os.cpu_count() or 1)))
        advanced_layout.addRow(self.ui.labelParallelJobs, self.ui.sbParallelJobs)

        self.ui.labelFallbackMode = QLabel(self.ui.groupBox_2)
        self.ui.cbFallbackMode = QComboBox(self.ui.groupBox_2)
        advanced_layout.addRow(self.ui.labelFallbackMode, self.ui.cbFallbackMode)

        self.ui.leDynamicOffset.setValidator(QDoubleValidator())
        self.ui.leVADSensitivity.setValidator(QDoubleValidator())
        self.ui.leVADHangover.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d+")))
        self.ui.leDynamicOffset.setText("6")
        self.ui.leVADSensitivity.setText("6")
        self.ui.leVADHangover.setText("120")
        self.ui.cbxDynamicThreshold.setChecked(False)
        self.ui.cbxVAD.setChecked(False)
        self.ui.sbParallelJobs.setValue(min(4, max(1, (os.cpu_count() or 1))))

        self._loading_presets = False
        self.ui.btnPresetSave.clicked.connect(self._on_save_preset)
        self.ui.btnPresetDelete.clicked.connect(self._on_delete_preset)
        self.ui.cbPresets.currentIndexChanged.connect(self._on_preset_selected)

    def _preset_file(self) -> str:
        base_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".audio_slicer")
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, "presets.json")

    def _load_presets(self):
        self._preset_path = self._preset_file()
        self._presets: dict[str, dict] = {}
        if os.path.exists(self._preset_path):
            try:
                with open(self._preset_path, "r", encoding="utf-8") as f:
                    self._presets = json.load(f)
            except Exception:
                self._presets = {}
        self._refresh_preset_combo()

    def _save_presets(self):
        with open(self._preset_path, "w", encoding="utf-8") as f:
            json.dump(self._presets, f, ensure_ascii=False, indent=2)

    def _refresh_preset_combo(self, selected_name: str | None = None):
        current_name = selected_name or self.ui.cbPresets.currentText()
        self._loading_presets = True
        self.ui.cbPresets.clear()
        self.ui.cbPresets.addItem(i18n.text("preset_select", self.current_language))
        for name in sorted(self._presets.keys()):
            self.ui.cbPresets.addItem(name)
        if current_name in self._presets:
            idx = self.ui.cbPresets.findText(current_name)
            if idx >= 0:
                self.ui.cbPresets.setCurrentIndex(idx)
        else:
            self.ui.cbPresets.setCurrentIndex(0)
        self._loading_presets = False

    def _collect_preset(self) -> dict:
        return {
            "threshold": self.ui.leThreshold.text(),
            "min_length": self.ui.leMinLen.text(),
            "min_interval": self.ui.leMinInterval.text(),
            "hop_size": self.ui.leHopSize.text(),
            "max_silence": self.ui.leMaxSilence.text(),
            "output_format": self._get_output_format(),
            "name_prefix": self.ui.leNamePrefix.text(),
            "name_suffix": self.ui.leNameSuffix.text(),
            "name_timestamp": self.ui.cbxNameTimestamp.isChecked(),
            "export_csv": self.ui.cbxExportCsv.isChecked(),
            "export_json": self.ui.cbxExportJson.isChecked(),
            "dynamic_enabled": self.ui.cbxDynamicThreshold.isChecked(),
            "dynamic_offset_db": self.ui.leDynamicOffset.text(),
            "vad_enabled": self.ui.cbxVAD.isChecked(),
            "vad_sensitivity_db": self.ui.leVADSensitivity.text(),
            "vad_hangover_ms": self.ui.leVADHangover.text(),
            "parallel_mode": self.ui.cbParallelMode.currentData(),
            "parallel_jobs": self.ui.sbParallelJobs.value(),
            "fallback_mode": self.ui.cbFallbackMode.currentData(),
        }

    def _apply_preset(self, data: dict):
        if not data:
            return
        if "threshold" in data:
            self.ui.leThreshold.setText(str(data["threshold"]))
        if "min_length" in data:
            self.ui.leMinLen.setText(str(data["min_length"]))
        if "min_interval" in data:
            self.ui.leMinInterval.setText(str(data["min_interval"]))
        if "hop_size" in data:
            self.ui.leHopSize.setText(str(data["hop_size"]))
        if "max_silence" in data:
            self.ui.leMaxSilence.setText(str(data["max_silence"]))
        if "output_format" in data:
            for btn in self.ui.outputFormatGroup.buttons():
                if btn.text() == data["output_format"]:
                    btn.setChecked(True)
                    break
        if "name_prefix" in data:
            self.ui.leNamePrefix.setText(str(data["name_prefix"]))
        if "name_suffix" in data:
            self.ui.leNameSuffix.setText(str(data["name_suffix"]))
        if "name_timestamp" in data:
            self.ui.cbxNameTimestamp.setChecked(bool(data["name_timestamp"]))
        if "export_csv" in data:
            self.ui.cbxExportCsv.setChecked(bool(data["export_csv"]))
        if "export_json" in data:
            self.ui.cbxExportJson.setChecked(bool(data["export_json"]))
        if "dynamic_enabled" in data:
            self.ui.cbxDynamicThreshold.setChecked(bool(data["dynamic_enabled"]))
        if "dynamic_offset_db" in data:
            self.ui.leDynamicOffset.setText(str(data["dynamic_offset_db"]))
        if "vad_enabled" in data:
            self.ui.cbxVAD.setChecked(bool(data["vad_enabled"]))
        if "vad_sensitivity_db" in data:
            self.ui.leVADSensitivity.setText(str(data["vad_sensitivity_db"]))
        if "vad_hangover_ms" in data:
            self.ui.leVADHangover.setText(str(data["vad_hangover_ms"]))
        if "parallel_mode" in data:
            idx = self.ui.cbParallelMode.findData(data["parallel_mode"])
            if idx >= 0:
                self.ui.cbParallelMode.setCurrentIndex(idx)
        if "parallel_jobs" in data:
            self.ui.sbParallelJobs.setValue(int(data["parallel_jobs"]))
        if "fallback_mode" in data:
            idx = self.ui.cbFallbackMode.findData(data["fallback_mode"])
            if idx >= 0:
                self.ui.cbFallbackMode.setCurrentIndex(idx)

    def _on_save_preset(self):
        name, ok = QInputDialog.getText(
            self,
            i18n.text("preset_save_title", self.current_language),
            i18n.text("preset_save_prompt", self.current_language),
        )
        if not ok or not name:
            return
        self._presets[name] = self._collect_preset()
        self._save_presets()
        self._refresh_preset_combo(name)

    def _on_delete_preset(self):
        name = self.ui.cbPresets.currentText()
        if name in self._presets:
            del self._presets[name]
            self._save_presets()
            self._refresh_preset_combo()

    def _on_preset_selected(self, index: int):
        if self._loading_presets:
            return
        name = self.ui.cbPresets.currentText()
        if name in self._presets:
            self._apply_preset(self._presets[name])

    def _init_language_selector(self):
        self.ui.cbLanguage.clear()
        for code, label in i18n.LANGUAGES.items():
            self.ui.cbLanguage.addItem(label, code)
        idx = self.ui.cbLanguage.findData(self.current_language)
        if idx >= 0:
            self.ui.cbLanguage.setCurrentIndex(idx)
        self.ui.cbLanguage.currentIndexChanged.connect(self._on_language_changed)

    def _refresh_parallel_mode_options(self):
        current = self.ui.cbParallelMode.currentData()
        self.ui.cbParallelMode.clear()
        self.ui.cbParallelMode.addItem(
            i18n.text("parallel_mode_single", self.current_language),
            "single",
        )
        self.ui.cbParallelMode.addItem(
            i18n.text("parallel_mode_thread", self.current_language),
            "thread",
        )
        self.ui.cbParallelMode.addItem(
            i18n.text("parallel_mode_process", self.current_language),
            "process",
        )
        if current is not None:
            idx = self.ui.cbParallelMode.findData(current)
            if idx >= 0:
                self.ui.cbParallelMode.setCurrentIndex(idx)

    def _refresh_fallback_mode_options(self):
        current = self.ui.cbFallbackMode.currentData()
        self.ui.cbFallbackMode.clear()
        self.ui.cbFallbackMode.addItem(
            i18n.text("fallback_mode_ask", self.current_language),
            "ask",
        )
        self.ui.cbFallbackMode.addItem(
            i18n.text("fallback_mode_ffmpeg_then_librosa", self.current_language),
            "ffmpeg_then_librosa",
        )
        self.ui.cbFallbackMode.addItem(
            i18n.text("fallback_mode_ffmpeg", self.current_language),
            "ffmpeg",
        )
        self.ui.cbFallbackMode.addItem(
            i18n.text("fallback_mode_librosa", self.current_language),
            "librosa",
        )
        self.ui.cbFallbackMode.addItem(
            i18n.text("fallback_mode_skip", self.current_language),
            "skip",
        )
        if current is not None:
            idx = self.ui.cbFallbackMode.findData(current)
            if idx >= 0:
                self.ui.cbFallbackMode.setCurrentIndex(idx)

    def _on_language_changed(self, index: int):
        code = self.ui.cbLanguage.itemData(index)
        if isinstance(code, str) and code:
            self.current_language = code
            self._apply_language()

    def _apply_language(self):
        self.setWindowTitle(i18n.text("window_title", self.current_language))
        self.ui.btnAddFiles.setText(i18n.text("add_files", self.current_language))
        self.ui.btnAbout.setText(i18n.text("about", self.current_language))
        self.ui.groupBox.setTitle(i18n.text("task_list", self.current_language))
        self.ui.btnRemove.setText(i18n.text("remove", self.current_language))
        self.ui.btnClearList.setText(i18n.text("clear_list", self.current_language))
        self.ui.groupBox_2.setTitle(i18n.text("settings", self.current_language))
        self.ui.label_2.setText(i18n.text("threshold", self.current_language))
        self.ui.label_3.setText(i18n.text("min_length", self.current_language))
        self.ui.label_4.setText(i18n.text("min_interval", self.current_language))
        self.ui.label_5.setText(i18n.text("hop_size", self.current_language))
        self.ui.label_6.setText(i18n.text("max_silence", self.current_language))
        self.ui.labelLanguage.setText(i18n.text("language", self.current_language))
        self.ui.btnPreviewSelection.setText(i18n.text("preview_selection", self.current_language))
        self.ui.label_7.setText(i18n.text("output_directory", self.current_language))
        self.ui.btnBrowse.setText(i18n.text("browse", self.current_language))
        self.ui.labelOutputFormat.setText(i18n.text("output_format", self.current_language))
        self.ui.cbxOpenOutuptDirectory.setText(i18n.text("open_output_directory", self.current_language))
        self.ui.btnStart.setText(
            i18n.text("slicing", self.current_language) if self.processing else i18n.text("start", self.current_language)
        )
        self.groupBoxPreview.setTitle(i18n.text("preview", self.current_language))
        if self._preview_pixmap_path:
            self._update_preview_pixmap()
        else:
            self.labelPreview.setText(i18n.text("preview_placeholder", self.current_language))
        self.ui.labelPreset.setText(i18n.text("presets", self.current_language))
        self.ui.btnPresetSave.setText(i18n.text("preset_save", self.current_language))
        self.ui.btnPresetDelete.setText(i18n.text("preset_delete", self.current_language))
        self.ui.labelNamePrefix.setText(i18n.text("name_prefix", self.current_language))
        self.ui.labelNameSuffix.setText(i18n.text("name_suffix", self.current_language))
        self.ui.labelNameTimestamp.setText(i18n.text("name_timestamp", self.current_language))
        self.ui.labelExportCsv.setText(i18n.text("export_csv", self.current_language))
        self.ui.labelExportJson.setText(i18n.text("export_json", self.current_language))
        self.ui.labelDynamicThreshold.setText(i18n.text("dynamic_threshold", self.current_language))
        self.ui.labelDynamicOffset.setText(i18n.text("dynamic_threshold_offset", self.current_language))
        self.ui.labelVAD.setText(i18n.text("vad", self.current_language))
        self.ui.labelVADSensitivity.setText(i18n.text("vad_sensitivity", self.current_language))
        self.ui.labelVADHangover.setText(i18n.text("vad_hangover", self.current_language))
        self.ui.labelParallelMode.setText(i18n.text("parallel_mode", self.current_language))
        self.ui.labelParallelJobs.setText(i18n.text("parallel_jobs", self.current_language))
        self.ui.labelFallbackMode.setText(i18n.text("fallback_mode", self.current_language))
        self.ui.settingsTabs.setTabText(0, i18n.text("settings_basic", self.current_language))
        self.ui.settingsTabs.setTabText(1, i18n.text("settings_advanced", self.current_language))
        self._refresh_parallel_mode_options()
        self._refresh_fallback_mode_options()
        self._refresh_preset_combo(self.ui.cbPresets.currentText())

    def _get_output_format(self) -> str:
        checked = self.ui.outputFormatGroup.checkedButton()
        if checked is None:
            return "wav"
        return checked.text()

    def _collect_processing_options(self) -> dict:
        return {
            "threshold_db": float(self.ui.leThreshold.text()),
            "min_length": int(self.ui.leMinLen.text()),
            "min_interval": int(self.ui.leMinInterval.text()),
            "hop_size": int(self.ui.leHopSize.text()),
            "max_silence": int(self.ui.leMaxSilence.text()),
            "dynamic_enabled": self.ui.cbxDynamicThreshold.isChecked(),
            "dynamic_offset_db": float(self.ui.leDynamicOffset.text()),
            "vad_enabled": self.ui.cbxVAD.isChecked(),
            "vad_sensitivity_db": float(self.ui.leVADSensitivity.text()),
            "vad_hangover_ms": int(self.ui.leVADHangover.text()),
            "parallel_mode": self.ui.cbParallelMode.currentData() or "single",
            "parallel_jobs": int(self.ui.sbParallelJobs.value()),
            "fallback_mode": self.ui.cbFallbackMode.currentData() or "ask",
            "name_prefix": self.ui.leNamePrefix.text(),
            "name_suffix": self.ui.leNameSuffix.text(),
            "name_timestamp": self.ui.cbxNameTimestamp.isChecked(),
            "export_csv": self.ui.cbxExportCsv.isChecked(),
            "export_json": self.ui.cbxExportJson.isChecked(),
            "output_dir": self.ui.leOutputDir.text() or None,
        }

    def _get_theme(self) -> str:
        color = self.palette().color(QPalette.Window)
        return "dark" if color.value() < 128 else "light"

    def _build_slice_analysis(self, slicer: Slicer, audio: np.ndarray):
        dynamic_enabled = self.ui.cbxDynamicThreshold.isChecked()
        vad_enabled = self.ui.cbxVAD.isChecked()
        rms_list = None
        if dynamic_enabled or vad_enabled:
            rms_list = slicer.get_rms_list(audio)
        dynamic_threshold_db = None
        if dynamic_enabled and rms_list is not None:
            dynamic_threshold_db = estimate_dynamic_threshold_db(
                rms_list,
                offset_db=float(self.ui.leDynamicOffset.text()),
            )
        vad_mask = None
        if vad_enabled and rms_list is not None:
            base_threshold = dynamic_threshold_db if dynamic_threshold_db is not None else slicer.threshold_db
            hangover_ms = int(self.ui.leVADHangover.text())
            hop_ms = int(self.ui.leHopSize.text())
            hangover_frames = 0
            if hop_ms > 0 and hangover_ms > 0:
                hangover_frames = max(1, int(round(hangover_ms / hop_ms)))
            vad_mask = build_vad_mask(
                rms_list,
                threshold_db=base_threshold,
                sensitivity_db=float(self.ui.leVADSensitivity.text()),
                hangover_frames=hangover_frames,
            )
        return rms_list, dynamic_threshold_db, vad_mask

    def _set_preview_image(self, image_path: str):
        self._preview_pixmap_path = image_path
        self._update_preview_pixmap()

    def _update_preview_pixmap(self):
        if not self._preview_pixmap_path:
            return
        pixmap = QPixmap(self._preview_pixmap_path)
        if pixmap.isNull():
            return
        target_size = self.labelPreview.size()
        scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.labelPreview.setPixmap(scaled)
        self.labelPreview.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_pixmap()

    def _on_preview_selection(self):
        if self.processing:
            self._warningProcessNotFinished()
            return
        item = self.ui.lwTaskList.currentItem()
        if item is None:
            QMessageBox.information(
                self,
                QApplication.applicationName(),
                i18n.text("preview_no_selection", self.current_language),
            )
            return
        filename = item.data(Qt.ItemDataRole.UserRole + 1)
        if not filename:
            return
        try:
            self._preview_with_file(filename)
        except Exception as exc:
            self._on_preview_error(filename, str(exc))

    def _preview_with_file(self, filename: str):
        audio, sr = soundfile.read(filename, dtype=np.float32)
        if len(audio.shape) > 1:
            audio = audio.T
        slicer = Slicer(
            sr=sr,
            threshold=float(self.ui.leThreshold.text()),
            min_length=int(self.ui.leMinLen.text()),
            min_interval=int(self.ui.leMinInterval.text()),
            hop_size=int(self.ui.leHopSize.text()),
            max_sil_kept=int(self.ui.leMaxSilence.text()),
        )
        rms_list, dynamic_threshold_db, vad_mask = self._build_slice_analysis(slicer, audio)
        sil_tags, total_frames, waveform_shape = slicer.get_slice_tags(
            audio,
            dynamic_threshold_db=dynamic_threshold_db,
            vad_mask=vad_mask,
            rms_list=rms_list,
        )
        preview = SlicingPreview(
            filename=filename,
            sil_tags=sil_tags,
            hop_size=int(self.ui.leHopSize.text()),
            total_frames=total_frames,
            waveform_shape=waveform_shape,
            theme=self._get_theme(),
            language=self.current_language,
        )
        preview_path = os.path.join(tempfile.gettempdir(), "audio_slicer_preview.png")
        preview.save_plot(preview_path)
        self._set_preview_image(preview_path)

    def _on_preview_error(self, filename: str, error: str):
        choice = self._show_fallback_dialog("preview_read_failed", filename, error)
        if choice == "ffmpeg":
            self._preview_with_ffmpeg(filename)
        elif choice == "librosa":
            self._preview_with_librosa(filename)

    def _show_fallback_dialog(self, prompt_key: str, filename: str, error: str) -> str:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(i18n.text("warning_title", self.current_language))
        msg.setText(
            i18n.text(prompt_key, self.current_language).format(
                file=filename,
                error=error,
            )
        )
        btn_ffmpeg = msg.addButton(
            i18n.text("preview_use_ffmpeg", self.current_language),
            QMessageBox.ActionRole,
        )
        btn_librosa = msg.addButton(
            i18n.text("preview_use_librosa", self.current_language),
            QMessageBox.ActionRole,
        )
        msg.addButton(QMessageBox.Cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_ffmpeg:
            return "ffmpeg"
        if clicked == btn_librosa:
            return "librosa"
        return "cancel"

    def _preview_with_ffmpeg(self, filename: str):
        ffmpeg_path = resolve_ffmpeg_path()
        if not ffmpeg_path:
            QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("ffmpeg_not_found", self.current_language),
            )
            return
        with tempfile.NamedTemporaryFile(
            prefix="audio_slicer_preview_",
            suffix=".wav",
            delete=False,
        ) as tmp:
            temp_path = tmp.name
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
            QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("ffmpeg_failed", self.current_language).format(
                    error=result.stderr.strip() or result.stdout.strip(),
                ),
            )
            return
        try:
            self._preview_with_file(temp_path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("read_failed", self.current_language).format(
                    file=filename,
                    error=str(exc),
                ),
            )

    def _preview_with_librosa(self, filename: str):
        try:
            import librosa
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("read_failed", self.current_language).format(
                    file=filename,
                    error=str(exc),
                ),
            )
            return
        try:
            audio, sr = librosa.load(filename, sr=None, mono=False)
            if audio.ndim > 1:
                audio_to_write = audio.T
            else:
                audio_to_write = audio
            with tempfile.NamedTemporaryFile(
                prefix="audio_slicer_preview_",
                suffix=".wav",
                delete=False,
            ) as tmp:
                temp_path = tmp.name
            soundfile.write(temp_path, audio_to_write, sr)
            self._preview_with_file(temp_path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("read_failed", self.current_language).format(
                    file=filename,
                    error=str(exc),
                ),
            )

    # Event Handlers
    def closeEvent(self, event):
        if self.processing:
            self._warningProcessNotFinished()
            event.ignore()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        has_wav = False
        for url in urls:
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1]
            if ext[1:].lower() in self.availableFormats:
                has_wav = True
                break
        if has_wav:
            event.accept()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1]
            if ext[1:].lower() not in self.availableFormats:
                continue
            item = QListWidgetItem()
            item.setSizeHint(QSize(200, 24))
            item.setText(QFileInfo(path).fileName())
            item.setData(Qt.ItemDataRole.UserRole + 1,
                         path)
            self.ui.lwTaskList.addItem(item)
