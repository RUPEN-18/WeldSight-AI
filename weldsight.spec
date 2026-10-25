# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all


# ============================================================
# COLLECT ULTRALYTICS
# ============================================================

ultra_datas, ultra_binaries, ultra_hidden = collect_all(
    "ultralytics"
)


# ============================================================
# COLLECT PYTORCH
# ============================================================

torch_datas, torch_binaries, torch_hidden = collect_all(
    "torch"
)


# ============================================================
# COLLECT TORCHVISION
# ============================================================

vision_datas, vision_binaries, vision_hidden = collect_all(
    "torchvision"
)


# ============================================================
# ANALYSIS
# ============================================================

a = Analysis(

    ["app/desktop_app.py"],

    pathex=["."],

    binaries=(
        ultra_binaries
        + torch_binaries
        + vision_binaries
    ),

    datas=[
        (
            "models/weld_yolo26s_v1.pt",
            "models"
        )
    ]
    + ultra_datas
    + torch_datas
    + vision_datas,

    hiddenimports=(
        ultra_hidden
        + torch_hidden
        + vision_hidden
        + [
            "cv2",
            "numpy",
            "PIL",
            "reportlab",
            "PySide6",
            "PySide6.QtCore",
            "PySide6.QtGui",
            "PySide6.QtWidgets",
        ]
    ),

    hookspath=[],

    hooksconfig={},

    runtime_hooks=[],

    excludes=[],

    noarchive=False,
)


# ============================================================
# PYTHON PACKAGE
# ============================================================

pyz = PYZ(
    a.pure
)


# ============================================================
# SINGLE EXE
# ============================================================

exe = EXE(

    pyz,

    a.scripts,

    a.binaries,

    a.datas,

    [],

    name="WeldSightAI",

    debug=False,

    bootloader_ignore_signals=False,

    strip=False,

    upx=False,

    console=False,

    disable_windowed_traceback=False,

    argv_emulation=False,

    target_arch=None,

    codesign_identity=None,

    entitlements_file=None,
)   