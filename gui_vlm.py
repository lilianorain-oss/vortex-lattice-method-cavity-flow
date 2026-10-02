"""Launcher for the VLM Tkinter GUI. Run from repo root:  python gui_vlm.py"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.gui_app import main

if __name__ == "__main__":
    main()
