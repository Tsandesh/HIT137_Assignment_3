import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import threading
import queue
import os

from models import TextToImageModel, ImageClassificationModel
from oop_concepts import OOPConcepts, log_decorator, time_decorator
from utils import make_thumbnail, detect_device
import torch

# -------------------------
# Base window (simple)
# -------------------------
class BaseGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Tkinter AI GUI")
        self.geometry("1000x700")
        self.minsize(900, 650)

# -------------------------
# Main Application (multiple inheritance)
# AIGUI inherits from BaseGUI and OOPConcepts to demonstrate multiple inheritance
# -------------------------
class AIGUI(BaseGUI, OOPConcepts):
    def __init__(self):
        BaseGUI.__init__(self)
        OOPConcepts.__init__(self)
        self._device = detect_device(torch)
        self._models = {
            "Text-to-Image": TextToImageModel(),
            "Image Classification": ImageClassificationModel()
        }
        # which model is selected by dropdown
        self.model_var = tk.StringVar(value="Text-to-Image")

        # Threading & result queue
        self._result_queue = queue.Queue()
        self._worker_thread = None

        # store last displayed image so PhotoImage reference is kept
        self._last_photoimage = None

        # build UI
        self._build_menu()
        self._build_top()
        self._build_center()
        self._build_bottom()
        self._poll_results()  # start polling the queue for model results

    # -------------------------
    # Menu bar
    # -------------------------
    def _build_menu(self):
        mb = tk.Menu(self)
        # File
        filem = tk.Menu(mb, tearoff=0)
        filem.add_command(label="Exit", command=self.quit)
        mb.add_cascade(label="File", menu=filem)
        # Models
        modelm = tk.Menu(mb, tearoff=0)
        modelm.add_command(label="Load Selected Model", command=self.load_selected_model)
        modelm.add_command(label="Load All Models (may be large)", command=self.load_all_models)
        mb.add_cascade(label="Models", menu=modelm)
        # Help
        helpm = tk.Menu(mb, tearoff=0)
        helpm.add_command(label="About", command=self._show_about)
        mb.add_cascade(label="Help", menu=helpm)
        self.config(menu=mb)

    # -------------------------
    # Top row (model selection + load button)
    # -------------------------
    def _build_top(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=8, pady=6)

        ttk.Label(top_frame, text="Model Selection:").pack(side="left", padx=(4,8))
        model_menu = ttk.OptionMenu(top_frame, self.model_var, self.model_var.get(), *self._models.keys())
        model_menu.pack(side="left")

        load_btn = ttk.Button(top_frame, text="Load Model", command=self.load_selected_model)
        load_btn.pack(side="left", padx=8)

        self.load_status = ttk.Label(top_frame, text=f"Device: {self._device}")
        self.load_status.pack(side="right", padx=8)

    # -------------------------
    # Center area: Left input / Right output
    # -------------------------
    def _build_center(self):
        center = ttk.Frame(self)
        center.pack(expand=True, fill="both", padx=8, pady=4)

        # Left = User Input Section
        left = ttk.LabelFrame(center, text="User Input Section", padding=(6,6))
        left.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # radio buttons: text, image
        self.input_mode = tk.StringVar(value="Text")
        mode_frame = ttk.Frame(left)
        mode_frame.pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Text", variable=self.input_mode, value="Text", command=self._on_input_mode_change).pack(side="left")
        ttk.Radiobutton(mode_frame, text="Image", variable=self.input_mode, value="Image", command=self._on_input_mode_change).pack(side="left")

        ttk.Button(mode_frame, text="Browse", command=self._browse_input).pack(side="left", padx=6)

        # text input area
        self.input_text = ScrolledText(left, height=7)
        self.input_text.pack(fill="both", expand=False, pady=(6,0))

        # image thumbnail preview under input text
        self.thumbnail_label = ttk.Label(left, text="(no image selected)")
        self.thumbnail_label.pack(pady=6)

        # buttons row (Run Model 1 / Run Model 2 / Clear)
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=4)

        self.run1_btn = ttk.Button(btn_frame, text="Run Model 1", command=self.run_selected_model)
        self.run1_btn.pack(side="left", padx=4)

        self.run2_btn = ttk.Button(btn_frame, text="Run Model 2", command=self.run_alternate_model)
        self.run2_btn.pack(side="left", padx=4)

        ttk.Button(btn_frame, text="Clear", command=self._clear_input).pack(side="left", padx=4)


        # Right = Model Output Section
        right = ttk.LabelFrame(center, text="Model Output Section", padding=(6,6))
        right.pack(side="right", fill="both", expand=True, padx=4, pady=4)

        ttk.Label(right, text="Output Display:").pack(anchor="nw")
        # area that can display text and images: we'll use a ScrolledText for textual output,
        # and a Label below it for images.
        self.output_text = ScrolledText(right, height=12, state="normal")
        self.output_text.pack(fill="both", expand=True)

        self.output_image_label = ttk.Label(right, text="(no image output)")
        self.output_image_label.pack(pady=6)

    # -------------------------
    # Bottom area: Model Info & OOP explanation (two columns)
    # -------------------------
    def _build_bottom(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=6)

        info_frame = ttk.LabelFrame(bottom, text="Model Information & Explanation", padding=(6,6))
        info_frame.pack(fill="both", expand=True)

        left_info = ttk.Frame(info_frame)
        left_info.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        ttk.Label(left_info, text="Selected Model Info:").pack(anchor="nw")
        self.model_info_text = tk.Text(left_info, height=8, width=40, wrap="word")
        self.model_info_text.pack(fill="both", expand=True)

        right_info = ttk.Frame(info_frame)
        right_info.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        ttk.Label(right_info, text="OOP Concepts Explanation:").pack(anchor="nw")
        self.oop_explanation_text = tk.Text(right_info, height=8, width=40, wrap="word")
        self.oop_explanation_text.pack(fill="both", expand=True)

        # small notes line
        notes = ttk.Label(self, text="Notes: Extra notes, instructions, or references.")
        notes.pack(side="bottom", anchor="w", padx=8, pady=(0,8))

        # fill OOP explanation initially using the OOPConcepts.explanation method (overridden)
        self._refresh_oop_explanation()

