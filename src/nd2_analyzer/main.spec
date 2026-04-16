# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('ui.py', '.'),
        ('/opt/anaconda3/lib/python3.12/site-packages/cachier/version.info', 'cachier'),
        ('/Applications/Oliveira Lab Projects/nd2-timelapse-analyzer/.venv/lib/python3.9/site-packages/distributed/distributed.yaml', 'distributed'),
    ],
    hiddenimports=[
        'nd2', 
        'matplotlib', 
        'seaborn', 
        'numpy', 
        'cv2', 
        'imageio.v3', 
        'tensorflow', 
        'cachier', 
        'cellpose',
        'torch', 
        'torchvision',
        'dask', 
        'distributed'
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
app = BUNDLE(
    coll,
    name='main.app',
    icon='nd2_favicon.icns',
    bundle_identifier=None,
)
