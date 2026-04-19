"""
Professional Medical Imaging GUI - Knee OA Parameter Analysis System.
Production-grade PyQt6 application for automated osteoarthritis diagnostics.
"""

import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QScrollArea, QTabWidget,
    QProgressBar, QSplitter, QFrame, QGroupBox, QGridLayout,
    QSizePolicy, QMessageBox, QDialog, QTextEdit, QStatusBar,
    QSlider, QSpinBox, QComboBox, QCheckBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QMimeData, QPropertyAnimation,
    QEasingCurve, QRect, QPoint
)
from PyQt6.QtGui import (
    QPixmap, QImage, QFont, QPainter, QColor, QPen, QBrush,
    QLinearGradient, QDragEnterEvent, QDropEvent, QIcon, QPalette,
    QAction, QWheelEvent, QMouseEvent
)

# ── Project imports ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometric_analysis import GeometricAnalyzer
from src.report_generator import ReportGenerator

# ═══════════════════════════════════════════════════════════════════
# THEME & STYLE
# ═══════════════════════════════════════════════════════════════════

DARK_THEME = """
QMainWindow, QWidget { background-color: #0f172a; color: #e2e8f0; }
QGroupBox {
    background-color: #1e293b; border: 1px solid #334155;
    border-radius: 8px; margin-top: 12px; padding: 16px 12px 12px;
    font-weight: bold; color: #94a3b8;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 16px; padding: 0 8px;
    color: #60a5fa; font-size: 13px;
}
QPushButton {
    background-color: #1e40af; color: white; border: none;
    border-radius: 6px; padding: 8px 18px; font-weight: bold; font-size: 12px;
}
QPushButton:hover { background-color: #2563eb; }
QPushButton:pressed { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #334155; color: #64748b; }
QPushButton#dangerBtn { background-color: #dc2626; }
QPushButton#dangerBtn:hover { background-color: #ef4444; }
QPushButton#successBtn { background-color: #059669; }
QPushButton#successBtn:hover { background-color: #10b981; }
QLabel { color: #e2e8f0; }
QLabel#sectionTitle { color: #60a5fa; font-size: 14px; font-weight: bold; }
QLabel#metricValue { color: #f8fafc; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace; }
QLabel#metricLabel { color: #94a3b8; font-size: 11px; }
QLabel#statusGreen { color: #22c55e; font-weight: bold; }
QLabel#statusYellow { color: #f59e0b; font-weight: bold; }
QLabel#statusRed { color: #ef4444; font-weight: bold; }
QProgressBar {
    background-color: #1e293b; border: 1px solid #334155; border-radius: 4px;
    text-align: center; color: white; font-size: 10px; height: 18px;
}
QProgressBar::chunk { background-color: #2563eb; border-radius: 3px; }
QProgressBar#greenBar::chunk { background-color: #22c55e; }
QProgressBar#yellowBar::chunk { background-color: #f59e0b; }
QProgressBar#redBar::chunk { background-color: #ef4444; }
QTabWidget::pane { border: 1px solid #334155; background: #0f172a; border-radius: 6px; }
QTabBar::tab {
    background: #1e293b; color: #94a3b8; padding: 8px 20px;
    border: 1px solid #334155; border-bottom: none; border-top-left-radius: 6px;
    border-top-right-radius: 6px; margin-right: 2px;
}
QTabBar::tab:selected { background: #0f172a; color: #60a5fa; font-weight: bold; }
QScrollArea { border: none; background: transparent; }
QStatusBar { background: #1e293b; color: #94a3b8; border-top: 1px solid #334155; }
QFrame#separator { background-color: #334155; }
"""


# ═══════════════════════════════════════════════════════════════════
# WORKER THREAD — Background image processing
# ═══════════════════════════════════════════════════════════════════

class AnalysisWorker(QThread):
    """Processes images in a background thread."""
    progress = pyqtSignal(int, str)       # (percentage, status_text)
    image_done = pyqtSignal(dict)          # single image result
    all_done = pyqtSignal(list)            # all results
    error = pyqtSignal(str)

    def __init__(self, image_paths: List[str], model=None, device="cpu"):
        super().__init__()
        self.image_paths = image_paths
        self.model = model
        self.device = device
        self.analyzer = GeometricAnalyzer()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        results = []
        total = len(self.image_paths)

        for i, path in enumerate(self.image_paths):
            if self._cancel:
                break

            fname = os.path.basename(path)
            self.progress.emit(int((i / total) * 100), f"Processing: {fname} ({i+1}/{total})")

            try:
                result = self._process_single(path)
                results.append(result)
                self.image_done.emit(result)
            except Exception as e:
                self.error.emit(f"Error processing {fname}: {str(e)}")
                results.append({"filename": fname, "error": str(e)})

        self.progress.emit(100, "Analysis complete!")
        self.all_done.emit(results)

    def _process_single(self, path: str) -> Dict:
        fname = os.path.basename(path)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness
        sharpness = self.analyzer.calculate_sharpness(gray)

        # 2. JSW geometric
        jsw = self.analyzer.measure_jsw(gray)

        # 3. Contour analysis
        contour = self.analyzer.analyze_contours(gray)

        # 4. CNN inference (if model loaded)
        kl_grade = 0
        confidence = 0.0
        clinical_params = {}
        grade_probs = [0.0] * 5

        if self.model is not None:
            try:
                from src.data_loader import ImagePreprocessor
                preprocessor = ImagePreprocessor()
                img_np = preprocessor.preprocess(path)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
                img_tensor = img_tensor.to(self.device)

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(img_tensor)

                if isinstance(outputs, tuple):
                    logits, params = outputs
                    params_np = params.squeeze().cpu().numpy()
                    param_names = ["JSW", "Osteophytes", "Sclerosis", "Contour"]
                    clinical_params = {n: float(v) for n, v in zip(param_names, params_np)}
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                kl_grade = int(np.argmax(probs))
                confidence = float(probs[kl_grade])
                grade_probs = probs.tolist()
            except Exception as e:
                logger_msg = f"CNN inference failed for {fname}: {e}"

        return {
            "filename": fname,
            "filepath": path,
            "timestamp": datetime.now().isoformat(),
            "sharpness": sharpness,
            "jsw_geometric": jsw,
            "contour_analysis": contour,
            "kl_grade": kl_grade,
            "confidence": confidence,
            "grade_probabilities": grade_probs,
            "clinical_params": clinical_params,
        }


# ═══════════════════════════════════════════════════════════════════
# DROP ZONE WIDGET
# ═══════════════════════════════════════════════════════════════════

class DropZone(QFrame):
    """Animated drag-and-drop area for X-ray images."""
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)
        self._dragging = False
        self._pulse_opacity = 0.0
        self._pulse_dir = 1

        self.setStyleSheet("""
            DropZone {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border: 2px dashed #475569; border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon_label = QLabel("📁")
        icon_label.setFont(QFont("Segoe UI Emoji", 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        title = QLabel("Drag & Drop X-ray Images Here")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e2e8f0;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("or click to browse files")
        subtitle.setStyleSheet("color: #64748b; font-size: 12px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        formats = QLabel("Supported: JPG, JPEG, PNG, TIFF  •  Max: 50 MB")
        formats.setStyleSheet("color: #475569; font-size: 10px;")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(formats)

        # Pulse timer
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._animate_pulse)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select Knee X-ray Images",
                str(Path.home() / "Pictures"),
                "Medical Images (*.jpg *.jpeg *.png *.tiff *.tif)"
            )
            if files:
                self.files_dropped.emit(files)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._dragging = True
            self.setStyleSheet("""
                DropZone {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1e3a5f, stop:1 #0f172a);
                    border: 2px dashed #3b82f6; border-radius: 12px;
                }
            """)
            self._pulse_timer.start(50)

    def dragLeaveEvent(self, event):
        self._dragging = False
        self._pulse_timer.stop()
        self.setStyleSheet("""
            DropZone {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border: 2px dashed #475569; border-radius: 12px;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self._dragging = False
        self._pulse_timer.stop()
        self.dragLeaveEvent(None)

        valid_ext = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
        max_size = 50 * 1024 * 1024  # 50MB

        paths = []
        for url in event.mimeData().urls():
            fpath = url.toLocalFile()
            if Path(fpath).suffix.lower() in valid_ext:
                if os.path.getsize(fpath) <= max_size:
                    paths.append(fpath)
                else:
                    QMessageBox.warning(self, "File Too Large",
                        f"{os.path.basename(fpath)} exceeds 50MB limit.")

        if paths:
            self.files_dropped.emit(paths)

    def _animate_pulse(self):
        self._pulse_opacity += 0.05 * self._pulse_dir
        if self._pulse_opacity >= 1.0:
            self._pulse_dir = -1
        elif self._pulse_opacity <= 0.0:
            self._pulse_dir = 1


# ═══════════════════════════════════════════════════════════════════
# IMAGE VIEWER (Zoom/Pan)
# ═══════════════════════════════════════════════════════════════════

class ImageViewer(QLabel):
    """Zoomable, pannable image viewer for medical images."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 6px;")
        self.setText("No image loaded")
        self.setFont(QFont("Segoe UI", 12))

        self._pixmap = None
        self._zoom = 1.0
        self._pan_start = None
        self._offset = QPoint(0, 0)

    def set_image(self, path: str):
        self._pixmap = QPixmap(path)
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._render()

    def set_cv_image(self, cv_img: np.ndarray):
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            qimg = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = cv_img.shape
            qimg = QImage(cv_img.data, w, h, w * ch, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._render()

    def _render(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            int(self._pixmap.width() * self._zoom),
            int(self._pixmap.height() * self._zoom),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = self._zoom * factor
        if 0.5 <= new_zoom <= 3.0:
            self._zoom = new_zoom
            self._render()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan_start = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._pan_start:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._pan_start = None

    def get_zoom_percent(self) -> int:
        return int(self._zoom * 100)


# ═══════════════════════════════════════════════════════════════════
# METRIC CARD Widget
# ═══════════════════════════════════════════════════════════════════

class MetricCard(QFrame):
    """Styled card showing a single metric with label, value, and progress bar."""

    def __init__(self, title: str, icon: str = "📊"):
        super().__init__()
        self.setStyleSheet("""
            MetricCard {
                background: #1e293b; border: 1px solid #334155;
                border-radius: 8px; padding: 0px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        # Header
        header = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setFont(QFont("Segoe UI Emoji", 14))
        header.addWidget(icon_lbl)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")
        header.addWidget(self.title_label)
        header.addStretch()
        layout.addLayout(header)

        # Content area
        self.content_layout = QGridLayout()
        self.content_layout.setSpacing(6)
        layout.addLayout(self.content_layout)

        self._row = 0

    def add_metric(self, label: str, value: str, bar_value: int = -1, status: str = ""):
        """Add a metric row: label | value | optional progress bar."""
        lbl = QLabel(label)
        lbl.setObjectName("metricLabel")
        lbl.setFont(QFont("Segoe UI", 10))

        val = QLabel(value)
        val.setObjectName("metricValue")
        val.setFont(QFont("Consolas", 12, QFont.Weight.Bold))

        self.content_layout.addWidget(lbl, self._row, 0)
        self.content_layout.addWidget(val, self._row, 1)

        if bar_value >= 0:
            bar = QProgressBar()
            bar.setValue(min(100, max(0, bar_value)))
            bar.setFixedHeight(14)
            bar.setTextVisible(False)
            if bar_value >= 70:
                bar.setObjectName("greenBar")
            elif bar_value >= 30:
                bar.setObjectName("yellowBar")
            else:
                bar.setObjectName("redBar")
            self.content_layout.addWidget(bar, self._row, 2)

        if status:
            st_label = QLabel(status)
            if "NORMAL" in status or "EXCELLENT" in status or "✓" in status:
                st_label.setObjectName("statusGreen")
            elif "MILD" in status or "GOOD" in status or "ACCEPTABLE" in status:
                st_label.setObjectName("statusYellow")
            else:
                st_label.setObjectName("statusRed")
            st_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            self.content_layout.addWidget(st_label, self._row, 3)

        self._row += 1

    def clear_metrics(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._row = 0


# ═══════════════════════════════════════════════════════════════════
# THUMBNAIL STRIP
# ═══════════════════════════════════════════════════════════════════

class ThumbnailStrip(QScrollArea):
    """Horizontal scrollable thumbnail bar."""
    image_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFixedHeight(110)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: #1e293b; border: 1px solid #334155; border-radius: 6px;")

        self._container = QWidget()
        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self._layout.addStretch()
        self.setWidget(self._container)

        self._thumbs: List[QLabel] = []
        self._selected = -1

    def add_thumbnail(self, path: str, index: int):
        pix = QPixmap(path).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setFixedSize(84, 84)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("border: 2px solid #475569; border-radius: 4px; padding: 1px;")
        lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        lbl.setToolTip(os.path.basename(path))
        lbl.mousePressEvent = lambda e, idx=index: self._on_click(idx)

        self._layout.insertWidget(self._layout.count() - 1, lbl)
        self._thumbs.append(lbl)

    def _on_click(self, index: int):
        self._selected = index
        for i, t in enumerate(self._thumbs):
            if i == index:
                t.setStyleSheet("border: 3px solid #3b82f6; border-radius: 4px; padding: 0px;")
            else:
                t.setStyleSheet("border: 2px solid #475569; border-radius: 4px; padding: 1px;")
        self.image_selected.emit(index)

    def clear_all(self):
        for t in self._thumbs:
            t.deleteLater()
        self._thumbs.clear()
        self._selected = -1


# ═══════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════

class KneeOAMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knee Osteoarthritis Parameter Analysis System")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # State
        self._image_paths: List[str] = []
        self._results: List[Dict] = []
        self._current_idx = -1
        self._model = None
        self._device = "cpu"
        self._worker: Optional[AnalysisWorker] = None
        self._report_gen = ReportGenerator()

        self._try_load_model()
        self._build_ui()

        self.statusBar().showMessage("Ready — Drop X-ray images to begin")

    # ── Model Loading ─────────────────────────────────────────
    def _try_load_model(self):
        model_path = PROJECT_ROOT / "models" / "best_model.pt"
        if model_path.exists():
            try:
                from src.model import create_model
                from configs.config import model_config
                self._model = create_model(model_config)
                state = torch.load(str(model_path), map_location="cpu")
                self._model.load_state_dict(state)
                self._model.eval()
                self.statusBar().showMessage("✅ CNN model loaded successfully")
            except Exception as e:
                self._model = None
        else:
            self._model = None

    # ── UI Construction ───────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Splitter: Left (images) | Right (results)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # ── LEFT PANEL ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 6, 12)
        left_layout.setSpacing(8)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        left_layout.addWidget(self.drop_zone)

        # Image viewer
        self.viewer = ImageViewer()
        self.viewer.hide()
        left_layout.addWidget(self.viewer, stretch=1)

        # Thumbnails
        self.thumb_strip = ThumbnailStrip()
        self.thumb_strip.image_selected.connect(self._on_thumb_selected)
        self.thumb_strip.hide()
        left_layout.addWidget(self.thumb_strip)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.btn_add = QPushButton("📁 Add Images")
        self.btn_add.clicked.connect(self._browse_files)
        ctrl_layout.addWidget(self.btn_add)

        self.btn_reprocess = QPushButton("🔄 Reprocess")
        self.btn_reprocess.setEnabled(False)
        self.btn_reprocess.clicked.connect(self._start_analysis)
        ctrl_layout.addWidget(self.btn_reprocess)

        self.btn_clear = QPushButton("🗑️ Clear All")
        self.btn_clear.setObjectName("dangerBtn")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self._clear_all)
        ctrl_layout.addWidget(self.btn_clear)

        ctrl_layout.addStretch()

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        ctrl_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.hide()
        ctrl_layout.addWidget(self.progress_bar)

        left_layout.addLayout(ctrl_layout)

        splitter.addWidget(left_panel)

        # ── RIGHT PANEL ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 12, 12, 12)
        right_layout.setSpacing(8)

        self.tabs = QTabWidget()

        # Tab 1: Individual Results
        self.individual_tab = QScrollArea()
        self.individual_tab.setWidgetResizable(True)
        self.individual_content = QWidget()
        self.individual_layout = QVBoxLayout(self.individual_content)
        self.individual_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.individual_layout.setSpacing(10)
        self.individual_tab.setWidget(self.individual_content)

        # Placeholder
        placeholder = QLabel("📋 Results will appear here after analysis")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #475569; font-size: 14px; padding: 60px;")
        self.individual_layout.addWidget(placeholder)

        self.tabs.addTab(self.individual_tab, "📊 Individual Results")

        # Tab 2: Averaged / Batch
        self.batch_tab = QScrollArea()
        self.batch_tab.setWidgetResizable(True)
        self.batch_content = QWidget()
        self.batch_layout = QVBoxLayout(self.batch_content)
        self.batch_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.batch_layout.setSpacing(10)
        self.batch_tab.setWidget(self.batch_content)
        self.tabs.addTab(self.batch_tab, "📈 Averaged Results")

        right_layout.addWidget(self.tabs)

        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        self.btn_json = QPushButton("💾 Save JSON")
        self.btn_json.setEnabled(False)
        self.btn_json.clicked.connect(lambda: self._export("json"))
        export_layout.addWidget(self.btn_json)

        self.btn_csv = QPushButton("📄 Save CSV")
        self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(lambda: self._export("csv"))
        export_layout.addWidget(self.btn_csv)

        self.btn_pdf = QPushButton("📑 Export PDF")
        self.btn_pdf.setObjectName("successBtn")
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(lambda: self._export("pdf"))
        export_layout.addWidget(self.btn_pdf)

        right_layout.addLayout(export_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([560, 840])

        root_layout.addWidget(splitter)

    # ── File Handling ─────────────────────────────────────────
    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Knee X-ray Images",
            str(Path.home() / "Pictures"),
            "Medical Images (*.jpg *.jpeg *.png *.tiff *.tif)"
        )
        if files:
            self._on_files_dropped(files)

    def _on_files_dropped(self, paths: List[str]):
        for p in paths:
            if p not in self._image_paths:
                self._image_paths.append(p)
                self.thumb_strip.add_thumbnail(p, len(self._image_paths) - 1)

        self.drop_zone.hide()
        self.viewer.show()
        self.thumb_strip.show()
        self.btn_clear.setEnabled(True)
        self.btn_reprocess.setEnabled(True)

        # Show first image
        if self._image_paths:
            self._current_idx = 0
            self.viewer.set_image(self._image_paths[0])

        self.progress_label.setText(f"{len(self._image_paths)} image(s) loaded")
        self.statusBar().showMessage(f"Loaded {len(paths)} image(s). Click Reprocess or drop more.")

        # Auto-start analysis
        self._start_analysis()

    def _on_thumb_selected(self, index: int):
        if 0 <= index < len(self._image_paths):
            self._current_idx = index
            self.viewer.set_image(self._image_paths[index])

            # Show corresponding results
            if index < len(self._results):
                self._display_individual(self._results[index])

    def _clear_all(self):
        self._image_paths.clear()
        self._results.clear()
        self._current_idx = -1
        self.thumb_strip.clear_all()
        self.thumb_strip.hide()
        self.viewer.hide()
        self.viewer.clear()
        self.drop_zone.show()
        self.btn_clear.setEnabled(False)
        self.btn_reprocess.setEnabled(False)
        self.btn_json.setEnabled(False)
        self.btn_csv.setEnabled(False)
        self.btn_pdf.setEnabled(False)
        self.progress_label.setText("")
        self._clear_layout(self.individual_layout)
        self._clear_layout(self.batch_layout)

        placeholder = QLabel("📋 Results will appear here after analysis")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #475569; font-size: 14px; padding: 60px;")
        self.individual_layout.addWidget(placeholder)

        self.statusBar().showMessage("Cleared. Drop images to begin.")

    # ── Analysis ──────────────────────────────────────────────
    def _start_analysis(self):
        if not self._image_paths:
            return

        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.btn_reprocess.setEnabled(False)

        self._worker = AnalysisWorker(self._image_paths, self._model, self._device)
        self._worker.progress.connect(self._on_progress)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_image_done(self, result: Dict):
        pass  # Could show live updates

    def _on_all_done(self, results: List[Dict]):
        self._results = results
        self.progress_bar.hide()
        self.btn_reprocess.setEnabled(True)
        self.btn_json.setEnabled(True)
        self.btn_csv.setEnabled(True)
        self.btn_pdf.setEnabled(True)

        # Show first result
        if results:
            self._display_individual(results[0])

        # Build batch tab
        if len(results) > 1:
            self._display_batch(results)

        count = len([r for r in results if "error" not in r])
        self.statusBar().showMessage(f"✅ Analysis complete — {count}/{len(results)} images processed successfully")
        self.progress_label.setText(f"{count} results ready")

    def _on_error(self, msg: str):
        self.statusBar().showMessage(f"⚠️ {msg}")

    # ── Results Display ───────────────────────────────────────
    def _display_individual(self, result: Dict):
        self._clear_layout(self.individual_layout)

        fname = result.get("filename", "Unknown")
        header = QLabel(f"Results for: {fname}")
        header.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        header.setStyleSheet("color: #f8fafc; padding: 4px 0;")
        self.individual_layout.addWidget(header)

        # ── Sharpness Card ──
        sh = result.get("sharpness", {})
        card_sharp = MetricCard("IMAGE QUALITY & SHARPNESS", "📊")
        card_sharp.add_metric("Sharpness Score:", f"{sh.get('score', 0):.1f} / 100",
                              bar_value=int(sh.get('score', 0)),
                              status=sh.get('status', 'N/A'))
        card_sharp.add_metric("Laplacian Variance:", f"{sh.get('laplacian_variance', 0):.1f}")
        card_sharp.add_metric("Edge Detection Score:", f"{sh.get('sobel_score', 0):.1f}")
        card_sharp.add_metric("Recommendation:", sh.get('recommendation', ''))
        self.individual_layout.addWidget(card_sharp)

        # ── JSW Card ──
        jsw = result.get("jsw_geometric", {})
        med = jsw.get("medial", {})
        lat = jsw.get("lateral", {})
        card_jsw = MetricCard("FEMUR-TIBIA DISTANCE (Joint Space Width)", "📏")
        card_jsw.add_metric("Measurement Points:", str(jsw.get("points_sampled", 0)))
        conf_pct = int(jsw.get("confidence", 0) * 100)
        card_jsw.add_metric("Detection Confidence:", f"{conf_pct}%", bar_value=conf_pct)
        card_jsw.add_metric("", "")  # spacer

        m_status = "✓ NORMAL" if med.get("mean", 0) >= 3.0 else "⚠ NARROWED"
        card_jsw.add_metric("Medial — Min / Max:", f"{med.get('min', 0):.1f} / {med.get('max', 0):.1f} mm")
        card_jsw.add_metric("Medial — Mean ± Std:", f"{med.get('mean', 0):.2f} ± {med.get('std', 0):.2f} mm",
                            status=m_status)

        l_status = "✓ NORMAL" if lat.get("mean", 0) >= 3.0 else "⚠ NARROWED"
        card_jsw.add_metric("Lateral — Min / Max:", f"{lat.get('min', 0):.1f} / {lat.get('max', 0):.1f} mm")
        card_jsw.add_metric("Lateral — Mean ± Std:", f"{lat.get('mean', 0):.2f} ± {lat.get('std', 0):.2f} mm",
                            status=l_status)
        self.individual_layout.addWidget(card_jsw)

        # ── Clinical Params Card (CNN) ──
        cp = result.get("clinical_params", {})
        if cp:
            card_cnn = MetricCard("CNN CLINICAL PARAMETERS", "🧠")

            # JSW param
            jsw_val = cp.get("JSW", 0)
            jsw_desc = "Healthy" if jsw_val > 0.7 else "Narrowed" if jsw_val > 0.3 else "Severe loss"
            card_cnn.add_metric("Joint Space Width:", f"{jsw_val:.3f}",
                                bar_value=int(jsw_val * 100), status=jsw_desc)

            ost_val = cp.get("Osteophytes", 0)
            ost_desc = "None" if ost_val < 0.2 else "Mild" if ost_val < 0.5 else "Severe"
            card_cnn.add_metric("Osteophytes:", f"{ost_val:.3f}",
                                bar_value=int(ost_val * 100), status=ost_desc)

            scl_val = cp.get("Sclerosis", 0)
            scl_desc = "Normal" if scl_val < 0.3 else "Mild" if scl_val < 0.6 else "Marked"
            card_cnn.add_metric("Subchondral Sclerosis:", f"{scl_val:.3f}",
                                bar_value=int(scl_val * 100), status=scl_desc)

            con_val = cp.get("Contour", 0)
            con_desc = "Smooth" if con_val < 0.3 else "Minimal changes" if con_val < 0.6 else "Deformed"
            card_cnn.add_metric("Bone Contour:", f"{con_val:.3f}",
                                bar_value=int(con_val * 100), status=con_desc)
            self.individual_layout.addWidget(card_cnn)

        # ── Contour Analysis Card ──
        ct = result.get("contour_analysis", {})
        if ct:
            card_contour = MetricCard("BONE CONTOUR CHANGES", "📐")
            card_contour.add_metric("Roughness Index:", f"{ct.get('roughness_index', 0):.1f} / 10.0")
            card_contour.add_metric("Shape Deviation:", f"{ct.get('shape_deviation', 0):.2f} mm")
            sev = ct.get("severity_grade", 0)
            card_contour.add_metric("Severity Grade:", f"Grade {sev}",
                                    status="✓ MINIMAL" if sev <= 1 else "⚠ MODERATE")
            self.individual_layout.addWidget(card_contour)

        # ── Overall Assessment ──
        grade_names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
        kl = result.get("kl_grade", 0)
        conf = result.get("confidence", 0)

        card_overall = MetricCard("OVERALL ASSESSMENT", "⚕️")
        card_overall.add_metric("Estimated KL Grade:",
                                f"Grade {kl} ({grade_names[kl]})",
                                bar_value=int(conf * 100),
                                status=f"Confidence: {conf:.0%}")

        # Grade probability bars
        probs = result.get("grade_probabilities", [0]*5)
        for g in range(5):
            pct = int(probs[g] * 100) if g < len(probs) else 0
            card_overall.add_metric(f"  Grade {g} ({grade_names[g]}):", f"{pct}%",
                                    bar_value=pct)
        self.individual_layout.addWidget(card_overall)

        # Disclaimer
        disclaimer = QLabel(
            "<i>⚠ This analysis is for research purposes only. "
            "Not a substitute for professional medical diagnosis.</i>"
        )
        disclaimer.setStyleSheet("color: #64748b; font-size: 10px; padding: 8px;")
        disclaimer.setWordWrap(True)
        self.individual_layout.addWidget(disclaimer)
        self.individual_layout.addStretch()

    def _display_batch(self, results: List[Dict]):
        self._clear_layout(self.batch_layout)
        import numpy as np

        valid = [r for r in results if "error" not in r]
        n = len(valid)
        if n == 0:
            return

        header = QLabel(f"Summary Statistics (N = {n} images)")
        header.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        header.setStyleSheet("color: #f8fafc; padding: 4px 0;")
        self.batch_layout.addWidget(header)

        # Gather arrays
        sharpness_scores = [r["sharpness"]["score"] for r in valid]
        jsw_medial = [r["jsw_geometric"]["medial"]["mean"] for r in valid]
        jsw_lateral = [r["jsw_geometric"]["lateral"]["mean"] for r in valid]
        kl_grades = [r["kl_grade"] for r in valid]
        confidences = [r["confidence"] for r in valid]

        card = MetricCard("AVERAGED PARAMETERS ACROSS ALL IMAGES", "📊")

        def fmt(vals): return f"{np.mean(vals):.1f} ± {np.std(vals):.1f}"
        def rng(vals): return f"{np.min(vals):.1f} – {np.max(vals):.1f}"

        acceptable = sum(1 for s in sharpness_scores if s >= 30)
        card.add_metric("Sharpness Average:", fmt(sharpness_scores),
                        bar_value=int(np.mean(sharpness_scores)))
        card.add_metric("Sharpness Range:", rng(sharpness_scores))
        card.add_metric("Acceptable Images:", f"{acceptable}/{n} ({acceptable/n:.0%})")
        card.add_metric("", "")

        card.add_metric("Medial JSW Average:", f"{np.mean(jsw_medial):.2f} ± {np.std(jsw_medial):.2f} mm")
        card.add_metric("Lateral JSW Average:", f"{np.mean(jsw_lateral):.2f} ± {np.std(jsw_lateral):.2f} mm")
        card.add_metric("", "")

        # Grade distribution
        for g in range(5):
            count = kl_grades.count(g)
            pct = count / n * 100
            names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
            card.add_metric(f"Grade {g} ({names[g]}):", f"{count} images ({pct:.0f}%)",
                            bar_value=int(pct))

        card.add_metric("", "")
        card.add_metric("Average KL Grade:", f"{np.mean(kl_grades):.1f} ± {np.std(kl_grades):.1f}")
        card.add_metric("Overall Confidence:", f"{np.mean(confidences):.1%} ± {np.std(confidences):.1%}",
                        bar_value=int(np.mean(confidences) * 100))

        self.batch_layout.addWidget(card)

        # CNN params averages
        if valid[0].get("clinical_params"):
            card_cnn = MetricCard("AVERAGED CNN CLINICAL PARAMETERS", "🧠")
            param_keys = ["JSW", "Osteophytes", "Sclerosis", "Contour"]
            for pk in param_keys:
                vals = [r["clinical_params"].get(pk, 0) for r in valid if r.get("clinical_params")]
                if vals:
                    card_cnn.add_metric(f"{pk}:", f"{np.mean(vals):.3f} ± {np.std(vals):.3f}",
                                        bar_value=int(np.mean(vals) * 100))
            self.batch_layout.addWidget(card_cnn)

        self.batch_layout.addStretch()

    # ── Export ────────────────────────────────────────────────
    def _export(self, fmt: str):
        if not self._results:
            return

        try:
            if fmt == "json":
                path = self._report_gen.export_json(self._results)
            elif fmt == "csv":
                path = self._report_gen.export_csv(self._results)
            elif fmt == "pdf":
                path = self._report_gen.export_pdf(self._results)
            else:
                return

            QMessageBox.information(self, "Export Successful",
                f"Report saved to:\n{path}")
            self.statusBar().showMessage(f"Exported {fmt.upper()} report: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ── Utilities ─────────────────────────────────────────────
    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME)

    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = KneeOAMainWindow()
    window.show()

    # Center on screen
    screen = app.primaryScreen().geometry()
    window.move(
        (screen.width() - window.width()) // 2,
        (screen.height() - window.height()) // 2
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
