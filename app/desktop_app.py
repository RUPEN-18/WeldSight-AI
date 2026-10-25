import sys
import uuid
from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSlider,
    QLineEdit,
    QMessageBox,
    QFrame,
    QSizePolicy,
    QScrollArea,
    QProgressBar,
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)


# ============================================================
# PATH HANDLING
# ============================================================

def resource_path(relative_path):
    """
    Used for resources bundled inside the EXE,
    such as the trained YOLO model.
    """

    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parents[1]

    return base_path / relative_path


def output_base_path():
    """
    Used for writable output files.

    Python mode:
        Weld_Inspection_AI/results/

    EXE mode:
        Folder containing WeldSightAI.exe/results/
    """

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parents[1]


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = resource_path(
    "models/weld_yolo26s_v1.pt"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

RESULTS_DIR = (
    output_base_path()
    / "results"
)

REPORTS_DIR = (
    RESULTS_DIR
    / "reports"
)

VIDEO_RESULTS_DIR = (
    RESULTS_DIR
    / "video_results"
)


REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIDEO_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VIDEO CONFIGURATION
# ============================================================

# Analyse approximately 8 frames every second
VIDEO_SAMPLE_FPS = 8


# ============================================================
# COLOURS
# ============================================================

NAVY = "#0B1F33"
NAVY_LIGHT = "#14314D"

BLUE = "#1565D8"
BLUE_HOVER = "#0E56BD"

BACKGROUND = "#F4F6F9"
CARD = "#FFFFFF"

TEXT = "#182230"
TEXT_SECONDARY = "#667085"
BORDER = "#E3E8EF"

GREEN = "#067647"
GREEN_BG = "#ECFDF3"

AMBER = "#B54708"
AMBER_BG = "#FFFAEB"

RED = "#B42318"
RED_BG = "#FEF3F2"


# ============================================================
# SEVERITY
# ============================================================

SEVERITY_ORDER = {
    "NORMAL": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def calculate_severity(
    box,
    image_width,
    image_height,
    class_name,
):

    if class_name == "Good Welding":
        return "NORMAL"

    x1, y1, x2, y2 = box

    defect_width = max(
        0,
        x2 - x1
    )

    defect_height = max(
        0,
        y2 - y1
    )

    defect_area = (
        defect_width
        * defect_height
    )

    image_area = (
        image_width
        * image_height
    )

    if image_area <= 0:
        return "LOW"

    ratio = (
        defect_area
        / image_area
    )

    if ratio < 0.02:
        return "LOW"

    elif ratio < 0.06:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# KPI CARD
# ============================================================

class MetricCard(QFrame):

    def __init__(
        self,
        title,
        value="0",
        accent=BLUE,
    ):

        super().__init__()

        self.setObjectName(
            "metricCard"
        )

        self.setMinimumHeight(
            92
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            18,
            14,
            18,
            14,
        )

        layout.setSpacing(
            3
        )

        self.title_label = QLabel(
            title
        )

        self.title_label.setStyleSheet(
            f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-weight: 700;
            """
        )

        self.value_label = QLabel(
            value
        )

        self.value_label.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 27px;
            font-weight: 700;
            """
        )

        accent_line = QFrame()

        accent_line.setFixedHeight(
            3
        )

        accent_line.setStyleSheet(
            f"""
            background-color: {accent};
            border-radius: 1px;
            """
        )

        layout.addWidget(
            self.title_label
        )

        layout.addWidget(
            self.value_label
        )

        layout.addStretch()

        layout.addWidget(
            accent_line
        )

    def set_value(
        self,
        value,
    ):

        self.value_label.setText(
            str(value)
        )

    def set_title(
        self,
        title,
    ):

        self.title_label.setText(
            title
        )


# ============================================================
# SECTION CARD
# ============================================================

class SectionCard(QFrame):

    def __init__(self):

        super().__init__()

        self.setObjectName(
            "sectionCard"
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

class WeldInspectionApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "WeldSight AI | Weld Quality Inspection"
        )

        self.resize(
            1500,
            920,
        )

        self.setMinimumSize(
            1180,
            720,
        )

        # ----------------------------------------------------
        # LOAD YOLO MODEL
        # ----------------------------------------------------

        try:

            self.model = YOLO(
                str(MODEL_PATH)
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Model Loading Error",
                f"Unable to load the trained model.\n\n"
                f"Model path:\n{MODEL_PATH}\n\n"
                f"Error:\n{error}"
            )

            raise

        # ----------------------------------------------------
        # APPLICATION STATE
        # ----------------------------------------------------

        self.current_mode = None

        self.current_image = None

        self.current_video = None

        self.annotated_image = None

        self.video_first_frame = None

        self.video_output_path = None

        self.detections = []

        self.video_summary = {}

        self.overall_severity = (
            "NORMAL"
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.setup_ui()

        self.apply_global_style()


    # ========================================================
    # GLOBAL STYLE
    # ========================================================

    def apply_global_style(self):

        self.setStyleSheet(
            f"""

            QMainWindow {{
                background: {BACKGROUND};
            }}

            QWidget {{
                font-family: "Segoe UI";
                color: {TEXT};
                font-size: 13px;
            }}

            QFrame#sectionCard,
            QFrame#metricCard {{

                background: {CARD};

                border: 1px solid {BORDER};

                border-radius: 10px;
            }}

            QLineEdit {{

                background: white;

                border: 1px solid #D0D5DD;

                border-radius: 6px;

                padding: 8px 10px;

                color: {TEXT};
            }}

            QLineEdit:focus {{

                border: 1px solid {BLUE};
            }}

            QTableWidget {{

                background: white;

                border: none;

                selection-background-color: #EAF2FF;

                selection-color: {TEXT};
            }}

            QTableWidget::item {{

                padding: 7px;

                border-bottom: 1px solid #F2F4F7;
            }}

            QHeaderView::section {{

                background: #F8FAFC;

                color: #475467;

                border: none;

                border-bottom: 1px solid #EAECF0;

                padding: 10px;

                font-size: 11px;

                font-weight: 700;
            }}

            QScrollArea {{

                border: none;

                background: {BACKGROUND};
            }}

            QScrollBar:vertical {{

                background: transparent;

                width: 10px;
            }}

            QScrollBar::handle:vertical {{

                background: #CBD5E1;

                border-radius: 5px;

                min-height: 25px;
            }}

            QSlider::groove:horizontal {{

                background: #DCE3EA;

                height: 5px;

                border-radius: 2px;
            }}

            QSlider::sub-page:horizontal {{

                background: {BLUE};

                border-radius: 2px;
            }}

            QSlider::handle:horizontal {{

                background: white;

                border: 2px solid {BLUE};

                width: 15px;

                margin: -6px 0;

                border-radius: 8px;
            }}

            QProgressBar {{

                border: 1px solid #D0D5DD;

                border-radius: 5px;

                text-align: center;

                background: white;

                height: 18px;
            }}

            QProgressBar::chunk {{

                background-color: {BLUE};

                border-radius: 4px;
            }}

            """
        )


    # ========================================================
    # ROOT UI
    # ========================================================

    def setup_ui(self):

        root = QWidget()

        self.setCentralWidget(
            root
        )

        root_layout = QHBoxLayout(
            root
        )

        root_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        root_layout.setSpacing(
            0
        )

        root_layout.addWidget(
            self.create_sidebar()
        )

        root_layout.addWidget(
            self.create_workspace(),
            1,
        )


    # ========================================================
    # SIDEBAR
    # ========================================================

    def create_sidebar(self):

        sidebar = QFrame()

        sidebar.setFixedWidth(
            225
        )

        sidebar.setStyleSheet(
            f"""
            background: {NAVY};
            """
        )

        layout = QVBoxLayout(
            sidebar
        )

        layout.setContentsMargins(
            20,
            25,
            20,
            22,
        )

        layout.setSpacing(
            8
        )


        brand = QLabel(
            "WELDSIGHT"
        )

        brand.setStyleSheet(
            """
            color: white;
            font-size: 21px;
            font-weight: 800;
            letter-spacing: 1px;
            """
        )


        subtitle = QLabel(
            "AI INSPECTION"
        )

        subtitle.setStyleSheet(
            """
            color: #8EA8C2;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            """
        )


        layout.addWidget(
            brand
        )

        layout.addWidget(
            subtitle
        )

        layout.addSpacing(
            28
        )


        layout.addWidget(
            self.nav_button(
                "Inspection",
                True
            )
        )

        layout.addWidget(
            self.nav_button(
                "Reports"
            )
        )

        layout.addWidget(
            self.nav_button(
                "Model Information"
            )
        )


        layout.addStretch()


        status = QFrame()

        status.setStyleSheet(
            f"""
            background: {NAVY_LIGHT};
            border-radius: 8px;
            """
        )


        status_layout = QVBoxLayout(
            status
        )

        status_layout.setContentsMargins(
            13,
            13,
            13,
            13,
        )


        status_title = QLabel(
            "MODEL STATUS"
        )

        status_title.setStyleSheet(
            """
            color: #8EA8C2;
            font-size: 9px;
            font-weight: 700;
            """
        )


        online = QLabel(
            "●  Online"
        )

        online.setStyleSheet(
            """
            color: #6CE9A6;
            font-size: 12px;
            font-weight: 700;
            """
        )


        version = QLabel(
            "YOLO26s\nWeldSight V1"
        )

        version.setStyleSheet(
            """
            color: #D5E0EA;
            font-size: 11px;
            """
        )


        status_layout.addWidget(
            status_title
        )

        status_layout.addWidget(
            online
        )

        status_layout.addWidget(
            version
        )


        layout.addWidget(
            status
        )

        return sidebar


    def nav_button(
        self,
        text,
        active=False,
    ):

        button = QPushButton(
            text
        )

        button.setMinimumHeight(
            42
        )

        button.setCursor(
            Qt.PointingHandCursor
        )

        background = (
            "#173A5E"
            if active
            else "transparent"
        )

        color = (
            "white"
            if active
            else "#A9BED1"
        )

        button.setStyleSheet(
            f"""

            QPushButton {{

                background: {background};

                color: {color};

                border: none;

                border-radius: 7px;

                text-align: left;

                padding-left: 14px;

                font-weight: 600;
            }}

            QPushButton:hover {{

                background: #173A5E;

                color: white;
            }}

            """
        )

        return button


    # ========================================================
    # WORKSPACE
    # ========================================================

    def create_workspace(self):

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )


        content = QWidget()

        content.setStyleSheet(
            f"""
            background: {BACKGROUND};
            """
        )


        main = QVBoxLayout(
            content
        )

        main.setContentsMargins(
            28,
            22,
            28,
            25,
        )

        main.setSpacing(
            16
        )


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title_row = QHBoxLayout()

        title_area = QVBoxLayout()


        title = QLabel(
            "Weld Quality Inspection"
        )

        title.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 25px;
            font-weight: 700;
            """
        )


        subtitle = QLabel(
            "Image and video based AI weld defect analysis"
        )

        subtitle.setStyleSheet(
            f"""
            color: {TEXT_SECONDARY};
            font-size: 12px;
            """
        )


        title_area.addWidget(
            title
        )

        title_area.addWidget(
            subtitle
        )


        title_row.addLayout(
            title_area
        )

        title_row.addStretch()


        prototype = QLabel(
            "PROTOTYPE"
        )

        prototype.setStyleSheet(
            f"""
            color: {BLUE};
            background: #EAF2FF;
            border-radius: 12px;
            padding: 5px 11px;
            font-size: 10px;
            font-weight: 800;
            """
        )


        title_row.addWidget(
            prototype
        )


        main.addLayout(
            title_row
        )


        # ----------------------------------------------------
        # INSPECTION INFORMATION
        # ----------------------------------------------------

        info_card = SectionCard()

        info_layout = QVBoxLayout(
            info_card
        )

        info_layout.setContentsMargins(
            18,
            15,
            18,
            16,
        )


        header = QLabel(
            "Inspection Information"
        )

        header.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 14px;
            font-weight: 700;
            """
        )


        info_layout.addWidget(
            header
        )


        grid = QGridLayout()


        self.inspection_id = QLineEdit(
            self.generate_inspection_id()
        )

        self.inspection_id.setReadOnly(
            True
        )


        self.weld_id = QLineEdit()

        self.weld_id.setPlaceholderText(
            "Example: WJ-001"
        )


        self.date_field = QLineEdit(
            datetime.now().strftime(
                "%d %b %Y"
            )
        )

        self.date_field.setReadOnly(
            True
        )


        labels = [
            "INSPECTION ID",
            "WELD / JOINT ID",
            "INSPECTION DATE",
        ]


        for index, text in enumerate(
            labels
        ):

            label = QLabel(
                text
            )

            label.setStyleSheet(
                f"""
                color: {TEXT_SECONDARY};
                font-size: 10px;
                font-weight: 700;
                """
            )

            grid.addWidget(
                label,
                0,
                index,
            )


        grid.addWidget(
            self.inspection_id,
            1,
            0,
        )

        grid.addWidget(
            self.weld_id,
            1,
            1,
        )

        grid.addWidget(
            self.date_field,
            1,
            2,
        )


        info_layout.addLayout(
            grid
        )


        main.addWidget(
            info_card
        )


        # ----------------------------------------------------
        # ACTION BAR
        # ----------------------------------------------------

        action_card = SectionCard()

        actions = QHBoxLayout(
            action_card
        )

        actions.setContentsMargins(
            15,
            12,
            15,
            12,
        )


        self.image_button = QPushButton(
            "Upload Image"
        )

        self.primary_button_style(
            self.image_button
        )

        self.image_button.clicked.connect(
            self.upload_image
        )


        self.video_button = QPushButton(
            "Upload Video"
        )

        self.primary_button_style(
            self.video_button
        )

        self.video_button.clicked.connect(
            self.upload_video
        )


        self.report_button = QPushButton(
            "Generate Report"
        )

        self.secondary_button_style(
            self.report_button
        )

        self.report_button.setEnabled(
            False
        )

        self.report_button.clicked.connect(
            self.generate_report
        )


        self.new_button = QPushButton(
            "New Inspection"
        )

        self.tertiary_button_style(
            self.new_button
        )

        self.new_button.clicked.connect(
            self.new_inspection
        )


        actions.addWidget(
            self.image_button
        )

        actions.addWidget(
            self.video_button
        )

        actions.addWidget(
            self.report_button
        )

        actions.addWidget(
            self.new_button
        )

        actions.addStretch()


        confidence_label = QLabel(
            "Confidence"
        )

        confidence_label.setStyleSheet(
            f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-weight: 600;
            """
        )


        actions.addWidget(
            confidence_label
        )


        self.confidence_slider = QSlider(
            Qt.Horizontal
        )

        self.confidence_slider.setRange(
            10,
            90,
        )

        # Slightly lower default to improve detection coverage
        self.confidence_slider.setValue(
            25
        )

        self.confidence_slider.setFixedWidth(
            120
        )


        self.confidence_value = QLabel(
            "25%"
        )


        self.confidence_slider.valueChanged.connect(
            lambda value:
            self.confidence_value.setText(
                f"{value}%"
            )
        )


        actions.addWidget(
            self.confidence_slider
        )

        actions.addWidget(
            self.confidence_value
        )


        main.addWidget(
            action_card
        )


        # ----------------------------------------------------
        # VIDEO PROGRESS
        # ----------------------------------------------------

        self.video_progress = QProgressBar()

        self.video_progress.setRange(
            0,
            100
        )

        self.video_progress.setValue(
            0
        )

        self.video_progress.hide()


        self.progress_text = QLabel(
            ""
        )

        self.progress_text.setStyleSheet(
            f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            """
        )

        self.progress_text.hide()


        main.addWidget(
            self.video_progress
        )

        main.addWidget(
            self.progress_text
        )


        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        metrics = QHBoxLayout()


        self.total_card = MetricCard(
            "TOTAL DEFECTS",
            "0",
            BLUE,
        )

        self.low_card = MetricCard(
            "LOW SEVERITY",
            "0",
            GREEN,
        )

        self.medium_card = MetricCard(
            "MEDIUM SEVERITY",
            "0",
            "#D97706",
        )

        self.high_card = MetricCard(
            "HIGH SEVERITY",
            "0",
            RED,
        )


        metrics.addWidget(
            self.total_card
        )

        metrics.addWidget(
            self.low_card
        )

        metrics.addWidget(
            self.medium_card
        )

        metrics.addWidget(
            self.high_card
        )


        main.addLayout(
            metrics
        )


        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        preview = QHBoxLayout()


        preview.addWidget(
            self.create_image_card(
                "Original / Source",
                False,
            )
        )


        preview.addWidget(
            self.create_image_card(
                "AI Analysis",
                True,
            )
        )


        main.addLayout(
            preview
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status_card = QFrame()

        status_layout = QHBoxLayout(
            self.status_card
        )


        self.status_indicator = QLabel(
            "●"
        )


        status_text = QVBoxLayout()


        self.status_title = QLabel(
            "Awaiting Inspection"
        )


        self.status_description = QLabel(
            "Upload an image or video to begin."
        )


        status_text.addWidget(
            self.status_title
        )

        status_text.addWidget(
            self.status_description
        )


        status_layout.addWidget(
            self.status_indicator
        )

        status_layout.addLayout(
            status_text
        )

        status_layout.addStretch()


        self.set_waiting_status()


        main.addWidget(
            self.status_card
        )


        # ----------------------------------------------------
        # DETECTION TABLE
        # ----------------------------------------------------

        table_card = SectionCard()

        table_layout = QVBoxLayout(
            table_card
        )


        table_title = QLabel(
            "Detection Results"
        )

        table_title.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 14px;
            font-weight: 700;
            """
        )


        self.table_description = QLabel(
            "Detected weld classifications"
        )

        self.table_description.setStyleSheet(
            f"""
            color: {TEXT_SECONDARY};
            font-size: 11px;
            """
        )


        self.table = QTableWidget()

        self.table.setColumnCount(
            5
        )

        self.table.setHorizontalHeaderLabels(
            [
                "CLASSIFICATION",
                "COUNT",
                "CONFIDENCE",
                "VISUAL SEVERITY",
                "STATUS",
            ]
        )


        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.table.verticalHeader().setVisible(
            False
        )

        self.table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        self.table.setSelectionBehavior(
            QTableWidget.SelectRows
        )

        self.table.setShowGrid(
            False
        )

        self.table.setMinimumHeight(
            190
        )


        table_layout.addWidget(
            table_title
        )

        table_layout.addWidget(
            self.table_description
        )

        table_layout.addWidget(
            self.table
        )


        main.addWidget(
            table_card
        )


        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        footer = QLabel(
            "AI visual inspection prototype • "
            "Engineering/NDT validation is required for final acceptance."
        )

        footer.setAlignment(
            Qt.AlignCenter
        )

        footer.setStyleSheet(
            """
            color: #98A2B3;
            font-size: 10px;
            """
        )


        main.addWidget(
            footer
        )


        scroll.setWidget(
            content
        )

        return scroll


    # ========================================================
    # IMAGE CARD
    # ========================================================

    def create_image_card(
        self,
        title,
        result,
    ):

        card = SectionCard()

        layout = QVBoxLayout(
            card
        )


        header = QLabel(
            title
        )

        header.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 13px;
            font-weight: 700;
            """
        )


        image = QLabel()

        image.setAlignment(
            Qt.AlignCenter
        )

        image.setMinimumHeight(
            300
        )

        image.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        image.setStyleSheet(
            f"""
            background: #F8FAFC;
            border: 1px solid {BORDER};
            border-radius: 7px;
            color: {TEXT_SECONDARY};
            """
        )


        if result:

            image.setText(
                "AI result will appear here"
            )

            self.result_image_label = (
                image
            )

        else:

            image.setText(
                "No source selected"
            )

            self.original_image_label = (
                image
            )


        layout.addWidget(
            header
        )

        layout.addWidget(
            image
        )

        return card


    # ========================================================
    # BUTTON STYLES
    # ========================================================

    def primary_button_style(
        self,
        button,
    ):

        button.setMinimumHeight(
            38
        )

        button.setCursor(
            Qt.PointingHandCursor
        )

        button.setStyleSheet(
            f"""

            QPushButton {{

                background: {BLUE};

                color: white;

                border: none;

                border-radius: 6px;

                padding: 8px 16px;

                font-weight: 700;
            }}

            QPushButton:hover {{

                background: {BLUE_HOVER};
            }}

            QPushButton:disabled {{

                background: #98A2B3;
            }}

            """
        )


    def secondary_button_style(
        self,
        button,
    ):

        button.setMinimumHeight(
            38
        )

        button.setStyleSheet(
            f"""

            QPushButton {{

                background: white;

                color: {BLUE};

                border: 1px solid #B5C9E8;

                border-radius: 6px;

                padding: 8px 16px;

                font-weight: 700;
            }}

            QPushButton:disabled {{

                color: #98A2B3;

                border: 1px solid #D0D5DD;

                background: #F9FAFB;
            }}

            """
        )


    def tertiary_button_style(
        self,
        button,
    ):

        button.setMinimumHeight(
            38
        )

        button.setStyleSheet(
            """

            QPushButton {

                background: white;

                color: #475467;

                border: 1px solid #D0D5DD;

                border-radius: 6px;

                padding: 8px 16px;
            }

            """
        )


    # ========================================================
    # INSPECTION ID
    # ========================================================

    def generate_inspection_id(self):

        date = datetime.now().strftime(
            "%Y%m%d"
        )

        code = (
            str(uuid.uuid4())
            [:4]
            .upper()
        )

        return (
            f"INS-{date}-{code}"
        )


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    def upload_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Weld Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )


        if not path:
            return


        self.current_mode = (
            "image"
        )

        self.current_image = (
            path
        )

        self.current_video = None

        self.video_output_path = None


        self.total_card.set_title(
            "TOTAL DEFECTS"
        )


        self.show_image(
            path,
            self.original_image_label,
        )


        self.run_image_detection(
            path
        )


    # ========================================================
    # IMAGE DETECTION
    # ========================================================

    def run_image_detection(
        self,
        path,
    ):

        threshold = (
            self.confidence_slider.value()
            / 100
        )


        self.set_processing_status(
            "Analysing Image",
            "YOLO26s is inspecting the selected weld image."
        )


        QApplication.processEvents()


        try:

            results = self.model.predict(
                source=path,
                conf=threshold,
                verbose=False,
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Detection Error",
                str(error)
            )

            return


        result = results[0]


        source = cv2.imread(
            path
        )


        if source is None:

            QMessageBox.critical(
                self,
                "Image Error",
                "Unable to open the selected image."
            )

            return


        height, width = (
            source.shape[:2]
        )


        self.annotated_image = (
            result.plot()
        )


        self.show_cv_image(
            self.annotated_image,
            self.result_image_label,
        )


        self.detections = []

        self.video_summary = {}

        self.overall_severity = (
            "NORMAL"
        )


        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )


                name = (
                    self.model.names[
                        class_id
                    ]
                )


                confidence = float(
                    box.conf[0]
                )


                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                severity = (
                    calculate_severity(
                        coordinates,
                        width,
                        height,
                        name,
                    )
                )


                self.detections.append(
                    {
                        "class": name,
                        "confidence": confidence,
                        "severity": severity,
                    }
                )


                if (
                    SEVERITY_ORDER[
                        severity
                    ]
                    >
                    SEVERITY_ORDER[
                        self.overall_severity
                    ]
                ):

                    self.overall_severity = (
                        severity
                    )


        self.update_image_metrics()

        self.update_image_table()

        self.update_final_status(
            self.detections
        )


        self.report_button.setEnabled(
            True
        )


    # ========================================================
    # VIDEO UPLOAD
    # ========================================================

    def upload_video(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Weld Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)",
        )


        if not path:
            return


        self.current_mode = (
            "video"
        )

        self.current_video = (
            path
        )

        self.current_image = None


        self.total_card.set_title(
            "DEFECT DETECTIONS"
        )


        self.process_video(
            path
        )


    # ========================================================
    # VIDEO PROCESSING
    # ========================================================

    def process_video(
        self,
        path,
    ):

        threshold = (
            self.confidence_slider.value()
            / 100
        )


        capture = cv2.VideoCapture(
            path
        )


        if not capture.isOpened():

            QMessageBox.critical(
                self,
                "Video Error",
                "Unable to open the selected video."
            )

            return


        fps = capture.get(
            cv2.CAP_PROP_FPS
        )


        if fps <= 0:

            fps = 25.0


        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )


        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        # ====================================================
        # 8 FPS SAMPLING
        # ====================================================

        sample_interval = max(
            1,
            round(
                fps
                / VIDEO_SAMPLE_FPS
            )
        )


        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        inspection_id = (
            self.inspection_id.text()
            .strip()
        )


        output_path = (
            VIDEO_RESULTS_DIR
            /
            f"{inspection_id}_{timestamp}_processed.mp4"
        )


        codec = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        writer = cv2.VideoWriter(
            str(output_path),
            codec,
            fps,
            (
                width,
                height
            ),
        )


        if not writer.isOpened():

            capture.release()

            QMessageBox.critical(
                self,
                "Video Output Error",
                "Unable to create the processed video."
            )

            return


        self.video_output_path = (
            output_path
        )


        self.video_progress.show()

        self.progress_text.show()

        self.video_progress.setValue(
            0
        )


        self.set_processing_status(
            "Analysing Video",
            f"Inspecting approximately {VIDEO_SAMPLE_FPS} frames per second."
        )


        self.image_button.setEnabled(
            False
        )

        self.video_button.setEnabled(
            False
        )

        self.new_button.setEnabled(
            False
        )

        self.report_button.setEnabled(
            False
        )


        QApplication.processEvents()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        class_summary = {}

        frame_index = 0

        analysed_frames = 0

        abnormal_frames = 0

        total_defect_detections = 0

        low_count = 0

        medium_count = 0

        high_count = 0


        self.overall_severity = (
            "NORMAL"
        )


        first_frame_saved = False

        representative_result = None


        try:

            while True:

                success, frame = (
                    capture.read()
                )


                if not success:
                    break


                if not first_frame_saved:

                    self.video_first_frame = (
                        frame.copy()
                    )


                    self.show_cv_image(
                        frame,
                        self.original_image_label,
                    )


                    first_frame_saved = (
                        True
                    )


                output_frame = (
                    frame.copy()
                )


                # ============================================
                # ANALYSE SAMPLED FRAME
                # ============================================

                if (
                    frame_index
                    % sample_interval
                    == 0
                ):

                    analysed_frames += 1


                    results = self.model.predict(
                        source=frame,
                        conf=threshold,
                        verbose=False,
                    )


                    result = results[0]


                    output_frame = (
                        result.plot()
                    )


                    frame_has_defect = (
                        False
                    )


                    if result.boxes is not None:

                        for box in result.boxes:

                            class_id = int(
                                box.cls[0]
                            )


                            class_name = (
                                self.model.names[
                                    class_id
                                ]
                            )


                            confidence = float(
                                box.conf[0]
                            )


                            coordinates = (
                                box.xyxy[0]
                                .cpu()
                                .numpy()
                            )


                            severity = (
                                calculate_severity(
                                    coordinates,
                                    width,
                                    height,
                                    class_name,
                                )
                            )


                            # --------------------------------
                            # CLASS AGGREGATION
                            # --------------------------------

                            if (
                                class_name
                                not in class_summary
                            ):

                                class_summary[
                                    class_name
                                ] = {

                                    "count": 0,

                                    "max_confidence": 0,

                                    "severity": "NORMAL",

                                }


                            class_summary[
                                class_name
                            ][
                                "count"
                            ] += 1


                            class_summary[
                                class_name
                            ][
                                "max_confidence"
                            ] = max(

                                class_summary[
                                    class_name
                                ][
                                    "max_confidence"
                                ],

                                confidence,

                            )


                            current_class_severity = (
                                class_summary[
                                    class_name
                                ][
                                    "severity"
                                ]
                            )


                            if (
                                SEVERITY_ORDER[
                                    severity
                                ]
                                >
                                SEVERITY_ORDER[
                                    current_class_severity
                                ]
                            ):

                                class_summary[
                                    class_name
                                ][
                                    "severity"
                                ] = severity


                            # --------------------------------
                            # DEFECT DETECTION
                            # --------------------------------

                            if (
                                class_name
                                != "Good Welding"
                            ):

                                frame_has_defect = (
                                    True
                                )


                                total_defect_detections += 1


                                if severity == "LOW":

                                    low_count += 1


                                elif severity == "MEDIUM":

                                    medium_count += 1


                                elif severity == "HIGH":

                                    high_count += 1


                                if (
                                    SEVERITY_ORDER[
                                        severity
                                    ]
                                    >
                                    SEVERITY_ORDER[
                                        self.overall_severity
                                    ]
                                ):

                                    self.overall_severity = (
                                        severity
                                    )


                    if frame_has_defect:

                        abnormal_frames += 1


                        representative_result = (
                            output_frame.copy()
                        )


                    self.show_cv_image(
                        output_frame,
                        self.result_image_label,
                    )


                # --------------------------------------------
                # WRITE FRAME TO OUTPUT VIDEO
                # --------------------------------------------

                writer.write(
                    output_frame
                )


                frame_index += 1


                if total_frames > 0:

                    progress = int(
                        frame_index
                        / total_frames
                        * 100
                    )

                else:

                    progress = 0


                self.video_progress.setValue(
                    min(
                        progress,
                        100
                    )
                )


                self.progress_text.setText(
                    f"Processing frame "
                    f"{frame_index:,}"
                    f" / "
                    f"{total_frames:,}"
                    f"   •   "
                    f"Sampling at ~{VIDEO_SAMPLE_FPS} FPS"
                )


                if (
                    frame_index
                    % 10
                    == 0
                ):

                    QApplication.processEvents()


        except Exception as error:

            QMessageBox.critical(
                self,
                "Video Processing Error",
                str(error)
            )


        finally:

            capture.release()

            writer.release()


            self.image_button.setEnabled(
                True
            )

            self.video_button.setEnabled(
                True
            )

            self.new_button.setEnabled(
                True
            )


        # ----------------------------------------------------
        # FINAL REPRESENTATIVE FRAME
        # ----------------------------------------------------

        if representative_result is not None:

            self.annotated_image = (
                representative_result
            )


        elif self.video_first_frame is not None:

            self.annotated_image = (
                self.video_first_frame.copy()
            )


        # ----------------------------------------------------
        # VIDEO SUMMARY
        # ----------------------------------------------------

        self.video_summary = {

            "total_frames": total_frames,

            "analysed_frames": analysed_frames,

            "abnormal_frames": abnormal_frames,

            "defect_detections": total_defect_detections,

            "low": low_count,

            "medium": medium_count,

            "high": high_count,

            "classes": class_summary,

            "source_fps": fps,

            "sample_fps": VIDEO_SAMPLE_FPS,

        }


        self.video_progress.setValue(
            100
        )


        self.progress_text.setText(
            f"Video inspection complete • "
            f"{analysed_frames} frames analysed at ~{VIDEO_SAMPLE_FPS} FPS."
        )


        self.update_video_metrics()

        self.update_video_table()

        self.update_video_status()


        self.report_button.setEnabled(
            True
        )


        QMessageBox.information(
            self,
            "Video Inspection Complete",

            "Video analysis completed successfully.\n\n"

            f"Sampling rate: {VIDEO_SAMPLE_FPS} FPS\n"

            f"Analysed frames: {analysed_frames}\n"

            f"Abnormal frames: {abnormal_frames}\n"

            f"Defect detections: {total_defect_detections}\n\n"

            "Processed video saved to:\n"

            f"{output_path}"
        )


    # ========================================================
    # IMAGE METRICS
    # ========================================================

    def update_image_metrics(self):

        defects = [

            detection

            for detection

            in self.detections

            if detection["class"]
            != "Good Welding"

        ]


        self.total_card.set_value(
            len(defects)
        )


        self.low_card.set_value(
            sum(
                detection["severity"]
                == "LOW"

                for detection

                in defects
            )
        )


        self.medium_card.set_value(
            sum(
                detection["severity"]
                == "MEDIUM"

                for detection

                in defects
            )
        )


        self.high_card.set_value(
            sum(
                detection["severity"]
                == "HIGH"

                for detection

                in defects
            )
        )


    # ========================================================
    # VIDEO METRICS
    # ========================================================

    def update_video_metrics(self):

        self.total_card.set_value(
            self.video_summary.get(
                "defect_detections",
                0
            )
        )


        self.low_card.set_value(
            self.video_summary.get(
                "low",
                0
            )
        )


        self.medium_card.set_value(
            self.video_summary.get(
                "medium",
                0
            )
        )


        self.high_card.set_value(
            self.video_summary.get(
                "high",
                0
            )
        )


    # ========================================================
    # IMAGE TABLE
    # ========================================================

    def update_image_table(self):

        self.table_description.setText(
            "Individual detections from the uploaded image"
        )


        self.table.setRowCount(
            len(
                self.detections
            )
        )


        for row, detection in enumerate(
            self.detections
        ):

            class_name = (
                detection[
                    "class"
                ]
            )


            confidence = (
                detection[
                    "confidence"
                ]
            )


            severity = (
                detection[
                    "severity"
                ]
            )


            status = (
                "PASS"
                if class_name
                == "Good Welding"
                else "REVIEW"
            )


            values = [

                class_name,

                "1",

                f"{confidence * 100:.1f}%",

                severity,

                status,

            ]


            self.populate_table_row(
                row,
                values,
                severity,
                status,
            )


    # ========================================================
    # VIDEO TABLE
    # ========================================================

    def update_video_table(self):

        classes = (
            self.video_summary.get(
                "classes",
                {}
            )
        )


        self.table_description.setText(
            "Aggregated detections from video frames sampled at "
            f"approximately {VIDEO_SAMPLE_FPS} FPS. "
            "Counts are frame detections, not unique physical defects."
        )


        self.table.setRowCount(
            len(classes)
        )


        for row, (
            class_name,
            stats,
        ) in enumerate(
            classes.items()
        ):

            severity = (
                stats[
                    "severity"
                ]
            )


            status = (
                "PASS"
                if class_name
                == "Good Welding"
                else "REVIEW"
            )


            values = [

                class_name,

                str(
                    stats[
                        "count"
                    ]
                ),

                (
                    f"{stats['max_confidence'] * 100:.1f}%"
                ),

                severity,

                status,

            ]


            self.populate_table_row(
                row,
                values,
                severity,
                status,
            )


    # ========================================================
    # TABLE ROW
    # ========================================================

    def populate_table_row(
        self,
        row,
        values,
        severity,
        status,
    ):

        for column, value in enumerate(
            values
        ):

            item = QTableWidgetItem(
                str(value)
            )


            item.setTextAlignment(
                Qt.AlignCenter
            )


            if column == 3:

                if severity == "LOW":

                    item.setForeground(
                        QColor(GREEN)
                    )


                elif severity == "MEDIUM":

                    item.setForeground(
                        QColor(AMBER)
                    )


                elif severity == "HIGH":

                    item.setForeground(
                        QColor(RED)
                    )


                else:

                    item.setForeground(
                        QColor(GREEN)
                    )


            if column == 4:

                if status == "PASS":

                    item.setForeground(
                        QColor(GREEN)
                    )

                else:

                    item.setForeground(
                        QColor(RED)
                    )


            self.table.setItem(
                row,
                column,
                item,
            )


    # ========================================================
    # WAITING STATUS
    # ========================================================

    def set_waiting_status(self):

        self.status_card.setStyleSheet(
            f"""
            background: #F8FAFC;
            border: 1px solid {BORDER};
            border-radius: 9px;
            """
        )


        self.status_indicator.setStyleSheet(
            """
            color: #98A2B3;
            font-size: 19px;
            """
        )


        self.status_title.setText(
            "Awaiting Inspection"
        )


        self.status_title.setStyleSheet(
            f"""
            color: {TEXT};
            font-weight: 700;
            """
        )


        self.status_description.setText(
            "Upload an image or video to begin AI analysis."
        )


    # ========================================================
    # PROCESSING STATUS
    # ========================================================

    def set_processing_status(
        self,
        title,
        description,
    ):

        self.status_card.setStyleSheet(
            """
            background: #EFF6FF;
            border: 1px solid #B2CCFF;
            border-radius: 9px;
            """
        )


        self.status_indicator.setStyleSheet(
            f"""
            color: {BLUE};
            font-size: 19px;
            """
        )


        self.status_title.setText(
            title
        )


        self.status_description.setText(
            description
        )


    # ========================================================
    # IMAGE FINAL STATUS
    # ========================================================

    def update_final_status(
        self,
        detections,
    ):

        defects = [

            detection

            for detection

            in detections

            if detection["class"]
            != "Good Welding"

        ]


        if not detections:

            self.set_status(
                AMBER_BG,
                "#FEDF89",
                AMBER,
                "No Classification Detected",
                "No prediction exceeded the selected confidence threshold."
            )

            return


        if not defects:

            self.set_status(
                GREEN_BG,
                "#ABEFC6",
                GREEN,
                "No Visual Defect Detected",
                "The current AI model detected no visible weld abnormality."
            )

            return


        self.set_defect_status()


    # ========================================================
    # VIDEO FINAL STATUS
    # ========================================================

    def update_video_status(self):

        abnormal_frames = (
            self.video_summary.get(
                "abnormal_frames",
                0
            )
        )


        analysed_frames = (
            self.video_summary.get(
                "analysed_frames",
                0
            )
        )


        if abnormal_frames == 0:

            self.set_status(
                GREEN_BG,
                "#ABEFC6",
                GREEN,
                "No Visual Defect Detected",

                f"{analysed_frames} sampled frames were analysed "
                f"at approximately {VIDEO_SAMPLE_FPS} FPS. "
                "No visible abnormality was detected."
            )

            return


        self.set_defect_status(
            extra=(
                f"{abnormal_frames} of "
                f"{analysed_frames} sampled frames "
                "contained abnormalities."
            )
        )


    # ========================================================
    # DEFECT STATUS
    # ========================================================

    def set_defect_status(
        self,
        extra="",
    ):

        if self.overall_severity == "LOW":

            background = GREEN_BG

            border = "#ABEFC6"

            color = GREEN


        elif self.overall_severity == "MEDIUM":

            background = AMBER_BG

            border = "#FEDF89"

            color = AMBER


        else:

            background = RED_BG

            border = "#FECDCA"

            color = RED


        description = (
            f"Highest visual severity: "
            f"{self.overall_severity}"
        )


        if extra:

            description += (
                f" • {extra}"
            )


        self.set_status(

            background,

            border,

            color,

            "Weld Requires Review",

            description,

        )


    # ========================================================
    # GENERIC STATUS
    # ========================================================

    def set_status(
        self,
        background,
        border,
        color,
        title,
        description,
    ):

        self.status_card.setStyleSheet(
            f"""
            background: {background};
            border: 1px solid {border};
            border-radius: 9px;
            """
        )


        self.status_indicator.setStyleSheet(
            f"""
            color: {color};
            font-size: 19px;
            """
        )


        self.status_title.setText(
            title
        )


        self.status_title.setStyleSheet(
            f"""
            color: {color};
            font-weight: 700;
            """
        )


        self.status_description.setText(
            description
        )


    # ========================================================
    # PDF REPORT
    # ========================================================

    def generate_report(self):

        if self.current_mode is None:

            QMessageBox.warning(
                self,
                "No Inspection",
                "Please perform an inspection first."
            )

            return


        inspection_id = (
            self.inspection_id.text()
            .strip()
        )


        weld_id = (
            self.weld_id.text()
            .strip()
            or "NOT PROVIDED"
        )


        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        pdf_path = (
            REPORTS_DIR
            /
            f"{inspection_id}_{timestamp}.pdf"
        )


        styles = (
            getSampleStyleSheet()
        )


        document = SimpleDocTemplate(

            str(pdf_path),

            pagesize=landscape(
                A4
            ),

            leftMargin=15 * mm,

            rightMargin=15 * mm,

            topMargin=12 * mm,

            bottomMargin=12 * mm,

        )


        elements = []


        # ----------------------------------------------------
        # REPORT TITLE
        # ----------------------------------------------------

        elements.append(
            Paragraph(
                "WeldSight AI Inspection Report",
                styles["Title"],
            )
        )


        elements.append(
            Spacer(
                1,
                8
            )
        )


        mode_name = (
            "IMAGE INSPECTION"
            if self.current_mode
            == "image"
            else "VIDEO INSPECTION"
        )


        info = [

            [
                "Inspection ID",
                inspection_id,
                "Weld / Joint ID",
                weld_id,
            ],

            [
                "Inspection Mode",
                mode_name,
                "AI Model",
                "YOLO26s WeldSight V1",
            ],

            [
                "Date",
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                ),
                "Highest Visual Severity",
                self.overall_severity,
            ],

        ]


        info_table = Table(
            info,

            colWidths=[
                40 * mm,
                55 * mm,
                45 * mm,
                55 * mm,
            ],
        )


        info_table.setStyle(
            TableStyle(
                [

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor(
                            "#D0D5DD"
                        ),
                    ),

                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.HexColor(
                            "#F2F4F7"
                        ),
                    ),

                    (
                        "BACKGROUND",
                        (2, 0),
                        (2, -1),
                        colors.HexColor(
                            "#F2F4F7"
                        ),
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),

                ]
            )
        )


        elements.append(
            info_table
        )


        elements.append(
            Spacer(
                1,
                12
            )
        )


        # ====================================================
        # IMAGE REPORT
        # ====================================================

        if self.current_mode == "image":

            detected_image_path = (
                REPORTS_DIR
                /
                f"{inspection_id}_{timestamp}_detected.jpg"
            )


            cv2.imwrite(
                str(
                    detected_image_path
                ),
                self.annotated_image,
            )


            images_table = Table(
                [

                    [

                        Paragraph(
                            "<b>Original Image</b>",
                            styles["Normal"],
                        ),

                        Paragraph(
                            "<b>AI Analysis</b>",
                            styles["Normal"],
                        ),

                    ],

                    [

                        PDFImage(
                            self.current_image,
                            width=90 * mm,
                            height=55 * mm,
                        ),

                        PDFImage(
                            str(
                                detected_image_path
                            ),
                            width=90 * mm,
                            height=55 * mm,
                        ),

                    ],

                ]
            )


            elements.append(
                images_table
            )


            report_rows = [

                [
                    "Classification",
                    "Confidence",
                    "Visual Severity",
                    "Status",
                ]

            ]


            for detection in self.detections:

                report_rows.append(
                    [

                        detection[
                            "class"
                        ],

                        (
                            f"{detection['confidence'] * 100:.1f}%"
                        ),

                        detection[
                            "severity"
                        ],

                        (
                            "PASS"
                            if detection[
                                "class"
                            ]
                            == "Good Welding"
                            else "REVIEW"
                        ),

                    ]
                )


        # ====================================================
        # VIDEO REPORT
        # ====================================================

        else:

            summary = (
                self.video_summary
            )


            elements.append(
                Paragraph(
                    "<b>Video Analysis Summary</b>",
                    styles["Heading2"],
                )
            )


            video_stats = [

                [
                    "Total Video Frames",
                    summary.get(
                        "total_frames",
                        0
                    ),

                    "Analysed Frames",
                    summary.get(
                        "analysed_frames",
                        0
                    ),
                ],

                [
                    "Abnormal Frames",
                    summary.get(
                        "abnormal_frames",
                        0
                    ),

                    "Defect Detections",
                    summary.get(
                        "defect_detections",
                        0
                    ),
                ],

                [
                    "Video Sampling Rate",
                    f"{VIDEO_SAMPLE_FPS} FPS",

                    "Confidence Threshold",
                    f"{self.confidence_slider.value()}%",
                ],

            ]


            summary_table = Table(

                video_stats,

                colWidths=[
                    45 * mm,
                    55 * mm,
                    45 * mm,
                    55 * mm,
                ],

            )


            summary_table.setStyle(
                TableStyle(
                    [

                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.HexColor(
                                "#D0D5DD"
                            ),
                        ),

                        (
                            "BACKGROUND",
                            (0, 0),
                            (0, -1),
                            colors.HexColor(
                                "#F2F4F7"
                            ),
                        ),

                        (
                            "BACKGROUND",
                            (2, 0),
                            (2, -1),
                            colors.HexColor(
                                "#F2F4F7"
                            ),
                        ),

                    ]
                )
            )


            elements.append(
                summary_table
            )


            elements.append(
                Spacer(
                    1,
                    12
                )
            )


            # -----------------------------------------------
            # REPRESENTATIVE VIDEO FRAME
            # -----------------------------------------------

            if self.annotated_image is not None:

                frame_path = (
                    REPORTS_DIR
                    /
                    f"{inspection_id}_{timestamp}_video_detection.jpg"
                )


                cv2.imwrite(
                    str(
                        frame_path
                    ),
                    self.annotated_image,
                )


                elements.append(
                    Paragraph(
                        "<b>Representative AI Detection Frame</b>",
                        styles["Heading2"],
                    )
                )


                elements.append(
                    PDFImage(
                        str(
                            frame_path
                        ),
                        width=110 * mm,
                        height=65 * mm,
                    )
                )


            report_rows = [

                [
                    "Classification",
                    "Frame Detections",
                    "Max Confidence",
                    "Highest Severity",
                    "Status",
                ]

            ]


            for (
                class_name,
                stats,
            ) in summary.get(
                "classes",
                {}
            ).items():

                report_rows.append(
                    [

                        class_name,

                        stats[
                            "count"
                        ],

                        (
                            f"{stats['max_confidence'] * 100:.1f}%"
                        ),

                        stats[
                            "severity"
                        ],

                        (
                            "PASS"
                            if class_name
                            == "Good Welding"
                            else "REVIEW"
                        ),

                    ]
                )


        # ----------------------------------------------------
        # FALLBACK IF NO DETECTIONS
        # ----------------------------------------------------

        if len(report_rows) == 1:

            if self.current_mode == "image":

                report_rows.append(
                    [
                        "No Detection",
                        "-",
                        "-",
                        "-",
                    ]
                )

            else:

                report_rows.append(
                    [
                        "No Detection",
                        "0",
                        "-",
                        "-",
                        "-",
                    ]
                )


        elements.append(
            Spacer(
                1,
                12
            )
        )


        report_table = Table(
            report_rows
        )


        report_table.setStyle(
            TableStyle(
                [

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#0B1F33"
                        ),
                    ),

                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor(
                            "#D0D5DD"
                        ),
                    ),

                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),

                ]
            )
        )


        elements.append(
            report_table
        )


        elements.append(
            Spacer(
                1,
                10
            )
        )


        elements.append(
            Paragraph(

                "<b>Prototype note:</b> "
                "Video counts represent detections across sampled frames "
                "and do not necessarily represent unique physical defects. "
                "LOW, MEDIUM and HIGH are visual severity estimates based "
                "on detected region size. This system does not replace "
                "qualified welding inspection or certified NDT procedures.",

                styles["Normal"],

            )
        )


        # ----------------------------------------------------
        # BUILD PDF
        # ----------------------------------------------------

        try:

            document.build(
                elements
            )


            QMessageBox.information(

                self,

                "Report Generated",

                "Inspection report generated successfully.\n\n"

                "Saved to:\n"

                f"{pdf_path}"

            )


        except Exception as error:

            QMessageBox.critical(
                self,
                "Report Error",
                str(error)
            )


    # ========================================================
    # NEW INSPECTION
    # ========================================================

    def new_inspection(self):

        self.current_mode = None

        self.current_image = None

        self.current_video = None

        self.video_output_path = None

        self.video_first_frame = None

        self.annotated_image = None

        self.detections = []

        self.video_summary = {}

        self.overall_severity = (
            "NORMAL"
        )


        self.inspection_id.setText(
            self.generate_inspection_id()
        )


        self.weld_id.clear()


        self.date_field.setText(
            datetime.now().strftime(
                "%d %b %Y"
            )
        )


        self.original_image_label.clear()

        self.original_image_label.setText(
            "No source selected"
        )


        self.result_image_label.clear()

        self.result_image_label.setText(
            "AI result will appear here"
        )


        self.table.setRowCount(
            0
        )


        self.total_card.set_title(
            "TOTAL DEFECTS"
        )


        self.total_card.set_value(
            0
        )

        self.low_card.set_value(
            0
        )

        self.medium_card.set_value(
            0
        )

        self.high_card.set_value(
            0
        )


        self.video_progress.hide()

        self.progress_text.hide()


        self.report_button.setEnabled(
            False
        )


        self.set_waiting_status()


    # ========================================================
    # DISPLAY IMAGE FROM FILE
    # ========================================================

    def show_image(
        self,
        path,
        label,
    ):

        pixmap = QPixmap(
            str(path)
        )


        if pixmap.isNull():

            label.setText(
                "Unable to display image"
            )

            return


        pixmap = pixmap.scaled(

            label.size(),

            Qt.KeepAspectRatio,

            Qt.SmoothTransformation,

        )


        label.setPixmap(
            pixmap
        )


    # ========================================================
    # DISPLAY OPENCV IMAGE
    # ========================================================

    def show_cv_image(
        self,
        image,
        label,
    ):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )


        height, width, channels = (
            rgb.shape
        )


        bytes_per_line = (
            channels
            * width
        )


        qimage = QImage(

            rgb.data,

            width,

            height,

            bytes_per_line,

            QImage.Format_RGB888,

        ).copy()


        pixmap = QPixmap.fromImage(
            qimage
        )


        pixmap = pixmap.scaled(

            label.size(),

            Qt.KeepAspectRatio,

            Qt.SmoothTransformation,

        )


        label.setPixmap(
            pixmap
        )


    # ========================================================
    # RESIZE
    # ========================================================

    def resizeEvent(
        self,
        event,
    ):

        super().resizeEvent(
            event
        )


        if (
            self.current_mode
            == "image"
            and
            self.current_image
        ):

            self.show_image(

                self.current_image,

                self.original_image_label,

            )


        elif (
            self.current_mode
            == "video"
            and
            self.video_first_frame
            is not None
        ):

            self.show_cv_image(

                self.video_first_frame,

                self.original_image_label,

            )


        if (
            self.annotated_image
            is not None
        ):

            self.show_cv_image(

                self.annotated_image,

                self.result_image_label,

            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )


    app.setApplicationName(
        "WeldSight AI"
    )


    app.setApplicationDisplayName(
        "WeldSight AI"
    )


    app.setFont(
        QFont(
            "Segoe UI",
            10,
        )
    )


    window = WeldInspectionApp()

    window.show()


    sys.exit(
        app.exec()
    )