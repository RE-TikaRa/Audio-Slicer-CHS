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
from audio_slicer.utils.slicer2 import Slicer, estimate_dynamic_threshold_db, build_vad_mask, get_rms
from audio_slicer.utils.processing import process_audio_file, resolve_ffmpeg_path

from audio_slicer.gui.Ui_MainWindow import Ui_MainWindow
from audio_slicer.utils.preview import SlicingPreview
from audio_slicer.modules import i18n

APP_VERSION = "1.5.0"


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
        self._preview_embed = False
        self._preview_window: QDialog | None = None
        self._preview_label: QLabel | None = None
        self._preview_zoom_label: QLabel | None = None
        self._preview_zoom_value: QLabel | None = None
        self._preview_zoom_slider: QSlider | None = None
        self._preview_original_pixmap: QPixmap | None = None
        self._preview_scroll_viewport: QWidget | None = None
        self._preview_pixmap_path: str | None = None
        self._style_sheet: str | None = None
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
        self.ui.btnRecommend.setEnabled(is_enabled)
        self.processing = processing

    def _init_extra_ui(self):
        if self._preview_embed:
            self._init_preview_panel()
        self._init_settings_tabs()
        self._init_main_splitter()
        self._init_recommend_controls()
        self._init_advanced_controls()
        self._apply_layout_style()
        self._apply_combo_popup_style()

    def _init_preview_panel(self):
        self.groupBoxPreview = QGroupBox(self)
        self.groupBoxPreview.setObjectName("groupBoxPreview")
        preview_layout = QVBoxLayout(self.groupBoxPreview)
        preview_layout.setContentsMargins(10, 12, 10, 10)
        preview_layout.setSpacing(8)
        self.labelPreview = QLabel(self.groupBoxPreview)
        self.labelPreview.setObjectName("labelPreview")
        self.labelPreview.setAlignment(Qt.AlignCenter)
        self.labelPreview.setWordWrap(True)
        self.labelPreview.setMinimumSize(240, 240)
        self.labelPreview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.labelPreview)
        self.ui.verticalLayout_3.removeWidget(self.ui.btnPreviewSelection)
        action_row = QHBoxLayout()
        action_row.addStretch()
        action_row.addWidget(self.ui.btnPreviewSelection)
        preview_layout.addLayout(action_row)
        self._preview_pixmap_path: str | None = None

    def _init_settings_tabs(self):
        self.ui.settingsTabs = QTabWidget(self.ui.groupBox_2)
        self.ui.tabBasic = QWidget(self.ui.settingsTabs)
        self.ui.tabAdvanced = QWidget(self.ui.settingsTabs)
        self.ui.basicFormLayout = self.ui.formLayout
        self.ui.advancedScroll = QScrollArea(self.ui.tabAdvanced)
        self.ui.advancedScroll.setWidgetResizable(True)
        self.ui.advancedContainer = QWidget(self.ui.advancedScroll)
        self.ui.advancedLayout = QVBoxLayout(self.ui.advancedContainer)
        self.ui.advancedLayout.setContentsMargins(6, 6, 6, 6)
        self.ui.advancedLayout.setSpacing(12)

        self.ui.verticalLayout_3.removeItem(self.ui.formLayout)
        self.ui.tabBasic.setLayout(self.ui.basicFormLayout)
        advanced_tab_layout = QVBoxLayout(self.ui.tabAdvanced)
        advanced_tab_layout.setContentsMargins(0, 0, 0, 0)
        advanced_tab_layout.addWidget(self.ui.advancedScroll)
        self.ui.advancedScroll.setWidget(self.ui.advancedContainer)
        self.ui.settingsTabs.addTab(self.ui.tabBasic, "")
        self.ui.settingsTabs.addTab(self.ui.tabAdvanced, "")
        self.ui.verticalLayout_3.insertWidget(0, self.ui.settingsTabs)

    def _init_recommend_controls(self):
        self.ui.labelRecommend = QLabel(self.ui.groupBox_2)
        self.ui.btnRecommend = QPushButton(self.ui.groupBox_2)
        self.ui.formLayout.insertRow(0, self.ui.labelRecommend, self.ui.btnRecommend)
        self.ui.btnRecommend.clicked.connect(self._on_recommend_params)

    def _init_main_splitter(self):
        self.ui.mainSplitter = QSplitter(Qt.Horizontal, self)
        self.ui.mainSplitter.setChildrenCollapsible(False)
        self.ui.leftSplitter = QSplitter(Qt.Vertical, self.ui.mainSplitter)
        self.ui.leftSplitter.setChildrenCollapsible(False)

        self.ui.horizontalLayout.removeWidget(self.ui.groupBox)
        self.ui.horizontalLayout.removeWidget(self.ui.groupBox_2)
        if self._preview_embed and self.groupBoxPreview:
            self.ui.horizontalLayout.removeWidget(self.groupBoxPreview)

        self.ui.leftSplitter.addWidget(self.ui.groupBox)
        if self._preview_embed and self.groupBoxPreview:
            self.ui.leftSplitter.addWidget(self.groupBoxPreview)
            self.ui.leftSplitter.setStretchFactor(0, 3)
            self.ui.leftSplitter.setStretchFactor(1, 2)
        else:
            self.ui.leftSplitter.setStretchFactor(0, 1)

        self.ui.mainSplitter.addWidget(self.ui.leftSplitter)
        self.ui.mainSplitter.addWidget(self.ui.groupBox_2)
        self.ui.mainSplitter.setStretchFactor(0, 3)
        self.ui.mainSplitter.setStretchFactor(1, 2)

        while self.ui.horizontalLayout.count():
            item = self.ui.horizontalLayout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self.ui.horizontalLayout.addWidget(self.ui.mainSplitter)

    def _apply_layout_style(self):
        self.ui.verticalLayout.setContentsMargins(12, 12, 12, 12)
        self.ui.verticalLayout.setSpacing(8)
        self.ui.verticalLayout_2.setContentsMargins(10, 12, 10, 10)
        self.ui.verticalLayout_2.setSpacing(8)
        self.ui.verticalLayout_3.setContentsMargins(10, 12, 10, 10)
        self.ui.verticalLayout_3.setSpacing(8)
        self.ui.formLayout.setHorizontalSpacing(10)
        self.ui.formLayout.setVerticalSpacing(8)
        self.ui.advancedPresetLayout.setHorizontalSpacing(10)
        self.ui.advancedPresetLayout.setVerticalSpacing(8)
        self.ui.advancedNamingLayout.setHorizontalSpacing(10)
        self.ui.advancedNamingLayout.setVerticalSpacing(8)
        self.ui.advancedDetectionLayout.setHorizontalSpacing(10)
        self.ui.advancedDetectionLayout.setVerticalSpacing(8)
        self.ui.advancedPerformanceLayout.setHorizontalSpacing(10)
        self.ui.advancedPerformanceLayout.setVerticalSpacing(8)
        if self.ui.advancedContainer:
            self.ui.advancedContainer.setMinimumWidth(1)
        self.ui.settingsTabs.setDocumentMode(True)
        self.ui.settingsTabs.setMovable(False)
        self.ui.groupBox_2.setMinimumWidth(320)
        if self._preview_embed and self.groupBoxPreview:
            self.groupBoxPreview.setMinimumHeight(240)
        self.setMinimumSize(1024, 640)
        self.resize(1180, 720)
        style = """
            QGroupBox {
                border: 1px solid rgba(120, 120, 120, 0.28);
                border-radius: 8px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid rgba(120, 120, 120, 0.22);
                border-radius: 8px;
            }
            QTabBar::tab {
                padding: 6px 12px;
                margin-right: 4px;
                border-radius: 8px;
            }
            QTabBar::tab:selected {
                background: rgba(59, 130, 246, 0.16);
                border-radius: 8px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
                padding: 4px 6px;
                border-radius: 8px;
            }
            QComboBox::drop-down {
                width: 20px;
                border: none;
            }
            QComboBox::down-arrow {
                width: 8px;
                height: 8px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid rgba(120, 120, 120, 0.28);
                border-radius: 8px;
                padding: 4px;
                background: palette(base);
                outline: 0;
            }
            QComboBox QListView {
                border-radius: 8px;
            }
            QComboBox QAbstractItemView::viewport {
                border-radius: 8px;
                background: palette(base);
            }
            QFrame#qt_ComboBox_Popup {
                border-radius: 8px;
                border: 1px solid rgba(120, 120, 120, 0.28);
                background: palette(base);
            }
            QFrame#qt_ComboBox_Popup QAbstractItemView {
                border-radius: 8px;
                border: none;
            }
            QComboBox QAbstractItemView::item {
                border-radius: 6px;
                padding: 4px 6px;
                margin: 2px;
            }
            QComboBox QAbstractItemView::item:selected {
                background: rgba(59, 130, 246, 0.16);
                border-radius: 6px;
            }
            QComboBox QAbstractItemView::item:hover {
                background: rgba(59, 130, 246, 0.10);
                border-radius: 6px;
            }
            QPushButton {
                padding: 6px 10px;
                border-radius: 8px;
            }
            QListWidget, QTableWidget, QTreeWidget, QScrollArea, QFrame {
                border-radius: 8px;
            }
            QListWidget::item, QTreeWidget::item, QTableWidget::item {
                border-radius: 6px;
                padding: 4px 6px;
            }
            QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
                background: rgba(59, 130, 246, 0.16);
                border-radius: 6px;
            }
            QProgressBar {
                border-radius: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 8px;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QMessageBox {
                border-radius: 8px;
            }
            QMessageBox QLabel {
                padding: 2px 0;
            }
            QMessageBox QPushButton {
                min-width: 80px;
            }
        """
        self._style_sheet = style
        self.setStyleSheet(style)
        if self._preview_window:
            self._preview_window.setStyleSheet(style)

    def _apply_combo_popup_style(self):
        for combo in self.findChildren(QComboBox):
            view = combo.view()
            if view is None:
                continue
            view.setFrameShape(QFrame.NoFrame)
            popup = view.window()
            if popup is None:
                continue
            popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            flags = popup.windowFlags()
            popup.setWindowFlags(
                flags
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.NoDropShadowWindowHint
            )
            if self._style_sheet:
                popup.setStyleSheet(self._style_sheet)

    def _init_advanced_controls(self):
        self.ui.groupAdvancedPresets = QGroupBox(self.ui.tabAdvanced)
        self.ui.groupAdvancedNaming = QGroupBox(self.ui.tabAdvanced)
        self.ui.groupAdvancedDetection = QGroupBox(self.ui.tabAdvanced)
        self.ui.groupAdvancedPerformance = QGroupBox(self.ui.tabAdvanced)

        self.ui.advancedPresetLayout = QFormLayout(self.ui.groupAdvancedPresets)
        self.ui.advancedNamingLayout = QFormLayout(self.ui.groupAdvancedNaming)
        self.ui.advancedDetectionLayout = QFormLayout(self.ui.groupAdvancedDetection)
        self.ui.advancedPerformanceLayout = QFormLayout(self.ui.groupAdvancedPerformance)

        self.ui.advancedLayout.addWidget(self.ui.groupAdvancedPresets)
        self.ui.advancedLayout.addWidget(self.ui.groupAdvancedNaming)
        self.ui.advancedLayout.addWidget(self.ui.groupAdvancedDetection)
        self.ui.advancedLayout.addWidget(self.ui.groupAdvancedPerformance)
        self.ui.advancedLayout.addStretch()

        for group in (
            self.ui.groupAdvancedPresets,
            self.ui.groupAdvancedNaming,
            self.ui.groupAdvancedDetection,
            self.ui.groupAdvancedPerformance,
        ):
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.ui.labelPreset = QLabel(self.ui.groupBox_2)
        self.ui.cbPresets = QComboBox(self.ui.groupBox_2)
        preset_view = QListView(self.ui.cbPresets)
        preset_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ui.cbPresets.setView(preset_view)
        self.ui.cbPresets.setMaxVisibleItems(8)
        self.ui.btnPresetSave = QPushButton(self.ui.groupBox_2)
        self.ui.btnPresetDelete = QPushButton(self.ui.groupBox_2)
        self.ui.btnPresetReset = QPushButton(self.ui.groupBox_2)
        preset_layout = QVBoxLayout()
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(6)
        preset_row.addWidget(self.ui.cbPresets, 1)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        button_row.addStretch()
        button_row.addWidget(self.ui.btnPresetSave)
        button_row.addWidget(self.ui.btnPresetDelete)
        button_row.addWidget(self.ui.btnPresetReset)
        preset_layout.addLayout(preset_row)
        preset_layout.addLayout(button_row)
        preset_widget = QWidget(self.ui.groupBox_2)
        preset_widget.setLayout(preset_layout)
        self.ui.advancedPresetLayout.addRow(self.ui.labelPreset, preset_widget)

        self.ui.labelNamePrefix = QLabel(self.ui.groupBox_2)
        self.ui.leNamePrefix = QLineEdit(self.ui.groupBox_2)
        self.ui.advancedNamingLayout.addRow(self.ui.labelNamePrefix, self.ui.leNamePrefix)

        self.ui.labelNameSuffix = QLabel(self.ui.groupBox_2)
        self.ui.leNameSuffix = QLineEdit(self.ui.groupBox_2)
        self.ui.advancedNamingLayout.addRow(self.ui.labelNameSuffix, self.ui.leNameSuffix)

        self.ui.labelNameTimestamp = QLabel(self.ui.groupBox_2)
        self.ui.cbxNameTimestamp = QCheckBox(self.ui.groupBox_2)
        self.ui.advancedNamingLayout.addRow(self.ui.labelNameTimestamp, self.ui.cbxNameTimestamp)

        self.ui.labelExportCsv = QLabel(self.ui.groupBox_2)
        self.ui.cbxExportCsv = QCheckBox(self.ui.groupBox_2)
        self.ui.advancedNamingLayout.addRow(self.ui.labelExportCsv, self.ui.cbxExportCsv)

        self.ui.labelExportJson = QLabel(self.ui.groupBox_2)
        self.ui.cbxExportJson = QCheckBox(self.ui.groupBox_2)
        self.ui.advancedNamingLayout.addRow(self.ui.labelExportJson, self.ui.cbxExportJson)

        self.ui.labelDynamicThreshold = QLabel(self.ui.groupBox_2)
        self.ui.cbxDynamicThreshold = QCheckBox(self.ui.groupBox_2)
        self.ui.advancedDetectionLayout.addRow(self.ui.labelDynamicThreshold, self.ui.cbxDynamicThreshold)

        self.ui.labelDynamicOffset = QLabel(self.ui.groupBox_2)
        self.ui.leDynamicOffset = QLineEdit(self.ui.groupBox_2)
        self.ui.leDynamicOffset.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.ui.advancedDetectionLayout.addRow(self.ui.labelDynamicOffset, self.ui.leDynamicOffset)

        self.ui.labelVAD = QLabel(self.ui.groupBox_2)
        self.ui.cbxVAD = QCheckBox(self.ui.groupBox_2)
        self.ui.advancedDetectionLayout.addRow(self.ui.labelVAD, self.ui.cbxVAD)

        self.ui.labelVADSensitivity = QLabel(self.ui.groupBox_2)
        self.ui.leVADSensitivity = QLineEdit(self.ui.groupBox_2)
        self.ui.leVADSensitivity.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.ui.advancedDetectionLayout.addRow(self.ui.labelVADSensitivity, self.ui.leVADSensitivity)

        self.ui.labelVADHangover = QLabel(self.ui.groupBox_2)
        self.ui.leVADHangover = QLineEdit(self.ui.groupBox_2)
        self.ui.leVADHangover.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.ui.advancedDetectionLayout.addRow(self.ui.labelVADHangover, self.ui.leVADHangover)

        self.ui.labelParallelMode = QLabel(self.ui.groupBox_2)
        self.ui.cbParallelMode = QComboBox(self.ui.groupBox_2)
        self.ui.advancedPerformanceLayout.addRow(self.ui.labelParallelMode, self.ui.cbParallelMode)

        self.ui.labelParallelJobs = QLabel(self.ui.groupBox_2)
        self.ui.sbParallelJobs = QSpinBox(self.ui.groupBox_2)
        self.ui.sbParallelJobs.setRange(1, max(1, (os.cpu_count() or 1)))
        self.ui.advancedPerformanceLayout.addRow(self.ui.labelParallelJobs, self.ui.sbParallelJobs)

        self.ui.labelFallbackMode = QLabel(self.ui.groupBox_2)
        self.ui.cbFallbackMode = QComboBox(self.ui.groupBox_2)
        self.ui.advancedPerformanceLayout.addRow(self.ui.labelFallbackMode, self.ui.cbFallbackMode)

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
        self.ui.btnPresetReset.clicked.connect(self._on_reset_presets)
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
        if not self._presets:
            self._presets = self._default_presets()
            self._save_presets()
        self._refresh_preset_combo()

    def _default_presets(self) -> dict:
        return {
            "默认（通用）": {
                "threshold": "-40",
                "min_length": "5000",
                "min_interval": "300",
                "hop_size": "10",
                "max_silence": "1000",
                "output_format": "wav",
                "name_prefix": "",
                "name_suffix": "",
                "name_timestamp": False,
                "export_csv": False,
                "export_json": False,
                "dynamic_enabled": True,
                "dynamic_offset_db": "6",
                "vad_enabled": True,
                "vad_sensitivity_db": "6",
                "vad_hangover_ms": "120",
                "parallel_mode": "thread",
                "parallel_jobs": min(4, max(1, (os.cpu_count() or 1))),
                "fallback_mode": "ffmpeg_then_librosa",
            },
            "人声（保守）": {
                "threshold": "-45",
                "min_length": "6000",
                "min_interval": "400",
                "hop_size": "10",
                "max_silence": "1500",
                "output_format": "wav",
                "name_prefix": "",
                "name_suffix": "",
                "name_timestamp": False,
                "export_csv": False,
                "export_json": False,
                "dynamic_enabled": True,
                "dynamic_offset_db": "5",
                "vad_enabled": True,
                "vad_sensitivity_db": "7",
                "vad_hangover_ms": "180",
                "parallel_mode": "thread",
                "parallel_jobs": min(4, max(1, (os.cpu_count() or 1))),
                "fallback_mode": "ffmpeg_then_librosa",
            },
            "人声（激进）": {
                "threshold": "-35",
                "min_length": "3000",
                "min_interval": "200",
                "hop_size": "10",
                "max_silence": "600",
                "output_format": "wav",
                "name_prefix": "",
                "name_suffix": "",
                "name_timestamp": False,
                "export_csv": False,
                "export_json": False,
                "dynamic_enabled": True,
                "dynamic_offset_db": "7",
                "vad_enabled": True,
                "vad_sensitivity_db": "5",
                "vad_hangover_ms": "80",
                "parallel_mode": "thread",
                "parallel_jobs": min(4, max(1, (os.cpu_count() or 1))),
                "fallback_mode": "ffmpeg_then_librosa",
            },
            "长音频": {
                "threshold": "-40",
                "min_length": "8000",
                "min_interval": "500",
                "hop_size": "20",
                "max_silence": "2000",
                "output_format": "flac",
                "name_prefix": "",
                "name_suffix": "",
                "name_timestamp": True,
                "export_csv": True,
                "export_json": False,
                "dynamic_enabled": True,
                "dynamic_offset_db": "6",
                "vad_enabled": True,
                "vad_sensitivity_db": "6",
                "vad_hangover_ms": "200",
                "parallel_mode": "thread",
                "parallel_jobs": min(4, max(1, (os.cpu_count() or 1))),
                "fallback_mode": "ffmpeg_then_librosa",
            },
            "播客/对白": {
                "threshold": "-42",
                "min_length": "4000",
                "min_interval": "250",
                "hop_size": "10",
                "max_silence": "1200",
                "output_format": "wav",
                "name_prefix": "",
                "name_suffix": "",
                "name_timestamp": False,
                "export_csv": True,
                "export_json": True,
                "dynamic_enabled": True,
                "dynamic_offset_db": "6",
                "vad_enabled": True,
                "vad_sensitivity_db": "6",
                "vad_hangover_ms": "150",
                "parallel_mode": "thread",
                "parallel_jobs": min(4, max(1, (os.cpu_count() or 1))),
                "fallback_mode": "ffmpeg_then_librosa",
            },
        }

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

    def _on_reset_presets(self):
        ret = QMessageBox.question(
            self,
            i18n.text("preset_reset_title", self.current_language),
            i18n.text("preset_reset_confirm", self.current_language),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self._presets = self._default_presets()
        self._save_presets()
        selected = next(iter(self._presets.keys()), None)
        self._refresh_preset_combo(selected)
        if selected:
            self._apply_preset(self._presets[selected])
        QMessageBox.information(
            self,
            i18n.text("preset_reset_done_title", self.current_language),
            i18n.text("preset_reset_done", self.current_language),
        )

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
        if self._preview_embed and self.groupBoxPreview:
            self.groupBoxPreview.setTitle(i18n.text("preview", self.current_language))
            if self._preview_pixmap_path:
                self._update_preview_pixmap()
            else:
                self.labelPreview.setText(i18n.text("preview_placeholder", self.current_language))
        if self._preview_window:
            self._preview_window.setWindowTitle(i18n.text("preview", self.current_language))
        if self._preview_zoom_label:
            self._preview_zoom_label.setText(i18n.text("preview_zoom", self.current_language))
        self.ui.labelPreset.setText(i18n.text("presets", self.current_language))
        self.ui.btnPresetSave.setText(i18n.text("preset_save", self.current_language))
        self.ui.btnPresetDelete.setText(i18n.text("preset_delete", self.current_language))
        self.ui.btnPresetReset.setText(i18n.text("preset_reset", self.current_language))
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
        self.ui.labelRecommend.setText(i18n.text("recommend_label", self.current_language))
        self.ui.btnRecommend.setText(i18n.text("recommend_button", self.current_language))
        self.ui.groupAdvancedPresets.setTitle(i18n.text("advanced_group_presets", self.current_language))
        self.ui.groupAdvancedNaming.setTitle(i18n.text("advanced_group_naming", self.current_language))
        self.ui.groupAdvancedDetection.setTitle(i18n.text("advanced_group_detection", self.current_language))
        self.ui.groupAdvancedPerformance.setTitle(i18n.text("advanced_group_performance", self.current_language))
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

    def _on_recommend_params(self):
        if self.processing:
            self._warningProcessNotFinished()
            return
        item = self.ui.lwTaskList.currentItem()
        if item is None and self.ui.lwTaskList.count() > 0:
            item = self.ui.lwTaskList.item(0)
        if item is None:
            QMessageBox.information(
                self,
                QApplication.applicationName(),
                i18n.text("recommend_no_selection", self.current_language),
            )
            return
        filename = item.data(Qt.ItemDataRole.UserRole + 1)
        if not filename:
            return
        audio, sr = self._read_audio_for_analysis(filename)
        if audio is None or sr is None:
            QMessageBox.warning(
                self,
                i18n.text("warning_title", self.current_language),
                i18n.text("recommend_failed", self.current_language),
            )
            return
        rec = self._compute_recommendations(audio, sr)
        msg = self._format_recommend_message(rec)
        ret = QMessageBox.question(
            self,
            i18n.text("recommend_title", self.current_language),
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if ret != QMessageBox.Yes:
            return
        self._apply_recommendations(rec)

    def _read_audio_for_analysis(self, filename: str):
        try:
            audio, sr = soundfile.read(filename, dtype=np.float32)
            return audio, sr
        except Exception as exc:
            choice = self._show_fallback_dialog("process_read_failed", filename, str(exc))
            if choice == "ffmpeg":
                return self._read_audio_with_ffmpeg(filename)
            if choice == "librosa":
                return self._read_audio_with_librosa(filename)
        return None, None

    def _read_audio_with_ffmpeg(self, filename: str):
        ffmpeg_path = resolve_ffmpeg_path()
        if not ffmpeg_path:
            return None, None
        with tempfile.NamedTemporaryFile(
            prefix="audio_slicer_recommend_",
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
                return None, None
            audio, sr = soundfile.read(temp_path, dtype=np.float32)
            return audio, sr
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _read_audio_with_librosa(self, filename: str):
        try:
            import librosa
        except Exception:
            return None, None
        try:
            audio, sr = librosa.load(filename, sr=None, mono=False)
            return audio, sr
        except Exception:
            return None, None

    def _compute_recommendations(self, audio: np.ndarray, sr: int) -> dict:
        if audio.ndim > 1:
            samples = audio.mean(axis=0)
        else:
            samples = audio
        duration_sec = len(samples) / max(sr, 1)
        hop_ms = 20 if duration_sec > 1200 else 10
        hop_length = max(1, int(sr * hop_ms / 1000))
        win_length = max(hop_length, min(int(sr * 0.03), 4 * hop_length))
        rms_list = get_rms(y=samples, frame_length=win_length, hop_length=hop_length).squeeze(0)
        rms_db = 20 * np.log10(np.clip(rms_list, a_min=1e-12, a_max=None))
        noise_floor = float(np.percentile(rms_db, 20))
        threshold_db = float(np.clip(noise_floor + 6.0, -80.0, -10.0))

        silent = rms_db < threshold_db
        silent_lengths = self._collect_run_lengths(silent)
        voice_lengths = self._collect_run_lengths(~silent)
        sil_ms = [int(length * hop_ms) for length in silent_lengths] if silent_lengths else []
        voice_ms = [int(length * hop_ms) for length in voice_lengths] if voice_lengths else []

        min_interval = int(np.clip(np.percentile(sil_ms, 50), 200, 1200)) if sil_ms else 300
        max_silence = int(np.clip(np.percentile(sil_ms, 90), 500, 5000)) if sil_ms else 1000
        min_length = int(np.clip(np.percentile(voice_ms, 20), 500, 8000)) if voice_ms else 5000
        min_interval = max(min_interval, hop_ms)
        max_silence = max(max_silence, hop_ms)
        min_length = max(min_length, min_interval)

        parallel_mode = "thread" if self.ui.lwTaskList.count() > 1 else "single"
        parallel_jobs = min(4, max(1, (os.cpu_count() or 1)))

        return {
            "threshold_db": round(threshold_db, 1),
            "min_length": min_length,
            "min_interval": min_interval,
            "hop_size": hop_ms,
            "max_silence": max_silence,
            "dynamic_enabled": True,
            "dynamic_offset_db": 6.0,
            "vad_enabled": True,
            "vad_sensitivity_db": 6.0,
            "vad_hangover_ms": 120,
            "parallel_mode": parallel_mode,
            "parallel_jobs": parallel_jobs,
            "fallback_mode": "ffmpeg_then_librosa",
        }

    def _collect_run_lengths(self, mask: np.ndarray) -> list[int]:
        lengths = []
        count = 0
        for value in mask:
            if value:
                count += 1
            elif count:
                lengths.append(count)
                count = 0
        if count:
            lengths.append(count)
        return lengths

    def _format_recommend_message(self, rec: dict) -> str:
        lang = self.current_language
        enabled = i18n.text("recommend_enabled", lang)
        disabled = i18n.text("recommend_disabled", lang)
        dyn_text = enabled if rec["dynamic_enabled"] else disabled
        vad_text = enabled if rec["vad_enabled"] else disabled
        parallel_text = {
            "single": i18n.text("parallel_mode_single", lang),
            "thread": i18n.text("parallel_mode_thread", lang),
            "process": i18n.text("parallel_mode_process", lang),
        }.get(rec["parallel_mode"], rec["parallel_mode"])
        fallback_text = {
            "ask": i18n.text("fallback_mode_ask", lang),
            "ffmpeg_then_librosa": i18n.text("fallback_mode_ffmpeg_then_librosa", lang),
            "ffmpeg": i18n.text("fallback_mode_ffmpeg", lang),
            "librosa": i18n.text("fallback_mode_librosa", lang),
            "skip": i18n.text("fallback_mode_skip", lang),
        }.get(rec["fallback_mode"], rec["fallback_mode"])
        return i18n.text("recommend_message", lang).format(
            threshold=rec["threshold_db"],
            min_length=rec["min_length"],
            min_interval=rec["min_interval"],
            hop_size=rec["hop_size"],
            max_silence=rec["max_silence"],
            dynamic_enabled=dyn_text,
            dynamic_offset=rec["dynamic_offset_db"],
            vad_enabled=vad_text,
            vad_sensitivity=rec["vad_sensitivity_db"],
            vad_hangover=rec["vad_hangover_ms"],
            parallel_mode=parallel_text,
            parallel_jobs=rec["parallel_jobs"],
            fallback_mode=fallback_text,
        )

    def _apply_recommendations(self, rec: dict):
        self.ui.leThreshold.setText(str(rec["threshold_db"]))
        self.ui.leMinLen.setText(str(rec["min_length"]))
        self.ui.leMinInterval.setText(str(rec["min_interval"]))
        self.ui.leHopSize.setText(str(rec["hop_size"]))
        self.ui.leMaxSilence.setText(str(rec["max_silence"]))
        self.ui.cbxDynamicThreshold.setChecked(bool(rec["dynamic_enabled"]))
        self.ui.leDynamicOffset.setText(str(rec["dynamic_offset_db"]))
        self.ui.cbxVAD.setChecked(bool(rec["vad_enabled"]))
        self.ui.leVADSensitivity.setText(str(rec["vad_sensitivity_db"]))
        self.ui.leVADHangover.setText(str(rec["vad_hangover_ms"]))
        idx = self.ui.cbParallelMode.findData(rec["parallel_mode"])
        if idx >= 0:
            self.ui.cbParallelMode.setCurrentIndex(idx)
        self.ui.sbParallelJobs.setValue(int(rec["parallel_jobs"]))
        idx = self.ui.cbFallbackMode.findData(rec["fallback_mode"])
        if idx >= 0:
            self.ui.cbFallbackMode.setCurrentIndex(idx)

    def _set_preview_image(self, image_path: str):
        if self._preview_embed:
            self._preview_pixmap_path = image_path
            self._update_preview_pixmap()
            return
        self._show_preview_window(image_path)

    def _update_preview_pixmap(self):
        if not self._preview_pixmap_path or not self._preview_embed:
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

    def eventFilter(self, obj, event):
        if obj is self._preview_scroll_viewport and event.type() == QEvent.Wheel:
            if self._preview_zoom_slider:
                delta = event.angleDelta().y()
                if delta != 0:
                    step = 10 if abs(delta) >= 120 else 5
                    value = self._preview_zoom_slider.value() + (step if delta > 0 else -step)
                    value = max(self._preview_zoom_slider.minimum(), min(self._preview_zoom_slider.maximum(), value))
                    self._preview_zoom_slider.setValue(value)
                return True
        return super().eventFilter(obj, event)

    def _show_preview_window(self, image_path: str):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        if self._preview_window is None:
            self._preview_window = QDialog(self)
            if self._style_sheet:
                self._preview_window.setStyleSheet(self._style_sheet)
            self._preview_window.setWindowTitle(i18n.text("preview", self.current_language))
            self._preview_window.setModal(False)
            layout = QVBoxLayout(self._preview_window)
            layout.setContentsMargins(10, 10, 10, 10)
            zoom_row = QHBoxLayout()
            self._preview_zoom_label = QLabel(i18n.text("preview_zoom", self.current_language), self._preview_window)
            self._preview_zoom_slider = QSlider(Qt.Horizontal, self._preview_window)
            self._preview_zoom_slider.setRange(25, 400)
            self._preview_zoom_slider.setValue(100)
            self._preview_zoom_value = QLabel("100%", self._preview_window)
            zoom_row.addWidget(self._preview_zoom_label)
            zoom_row.addWidget(self._preview_zoom_slider, 1)
            zoom_row.addWidget(self._preview_zoom_value)
            layout.addLayout(zoom_row)
            scroll = QScrollArea(self._preview_window)
            scroll.setWidgetResizable(True)
            self._preview_label = QLabel(scroll)
            self._preview_label.setAlignment(Qt.AlignCenter)
            self._preview_label.setScaledContents(False)
            scroll.setWidget(self._preview_label)
            layout.addWidget(scroll)
            self._preview_zoom_slider.valueChanged.connect(self._on_preview_zoom_changed)
            self._preview_scroll_viewport = scroll.viewport()
            if self._preview_scroll_viewport:
                self._preview_scroll_viewport.installEventFilter(self)
        else:
            self._preview_window.setWindowTitle(i18n.text("preview", self.current_language))
        self._preview_original_pixmap = pixmap
        if self._preview_zoom_slider:
            self._preview_zoom_slider.setValue(100)
        self._update_preview_zoom()
        target_width = min(1400, pixmap.width() + 40)
        target_height = min(900, pixmap.height() + 60)
        self._preview_window.resize(target_width, target_height)
        self._preview_window.show()
        self._preview_window.raise_()
        self._preview_window.activateWindow()

    def _on_preview_zoom_changed(self, value: int):
        if self._preview_zoom_value:
            self._preview_zoom_value.setText(f"{value}%")
        self._update_preview_zoom()

    def _update_preview_zoom(self):
        if not self._preview_label or not self._preview_original_pixmap or not self._preview_zoom_slider:
            return
        scale = self._preview_zoom_slider.value() / 100.0
        target_size = self._preview_original_pixmap.size() * scale
        scaled = self._preview_original_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._preview_label.setPixmap(scaled)

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
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

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
            try:
                self._preview_with_file(temp_path)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
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
