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

    # -------------------------
    # UI helpers
    # -------------------------
    def _show_about(self):
        messagebox.showinfo("About", "Tkinter AI GUI\nDemonstration of OOP concepts + Hugging Face models.")

    def _on_input_mode_change(self):
        mode = self.input_mode.get()
        if mode == "Text":
            self.input_text.config(state="normal")
        else:
            # keep text editable but user likely will use Browse
            self.input_text.config(state="normal")
        # reset thumbnail label if switching away
        if mode != "Image":
            self.thumbnail_label.config(text="(no image selected)")

    def _browse_input(self):
        mode = self.input_mode.get()
        if mode == "Text":
            # optional: let user choose a .txt file to load into text area
            path = filedialog.askopenfilename(title="Open text file", filetypes=[("Text files","*.txt"),("All files","*.*")])
            if path:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert("1.0", txt)
        else:
            path = filedialog.askopenfilename(title="Open image", filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.gif"),("All files","*.*")])
            if path:
                # show thumbnail
                try:
                    img = Image.open(path)
                    thumb = make_thumbnail(img)
                    photo = ImageTk.PhotoImage(thumb)
                    self._last_photoimage = photo
                    self.thumbnail_label.config(image=photo, text="")
                    # store selected path in an attribute
                    self._selected_input_path = path
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open image: {e}")

    def _clear_input(self):
        self.input_text.delete("1.0", tk.END)
        self.thumbnail_label.config(image="", text="(no image selected)")
        self.output_text.delete("1.0", tk.END)
        self.output_image_label.config(image="", text="(no image output)")
        self._last_photoimage = None

    # -------------------------
    # Model load/run actions
    # -------------------------
    def load_selected_model(self):
        name = self.model_var.get()
        model = self._models[name]
        self._append_output(f"Loading {name} on device {self._device} ...")
        # load in background
        t = threading.Thread(target=self._load_model_thread, args=(name,), daemon=True)
        t.start()

    def load_all_models(self):
        # load both in a thread
        self._append_output("Loading all models (this may take a while)...")
        t = threading.Thread(target=self._load_all_thread, daemon=True)
        t.start()

    def _load_model_thread(self, name):
        try:
            model = self._models[name]
            model.load(device=self._device)
            self._result_queue.put(("load_ok", f"{name} loaded."))
        except Exception as e:
            self._result_queue.put(("error", f"Error loading {name}: {e}"))

    def _load_all_thread(self):
        errors = []
        for name, model in self._models.items():
            try:
                model.load(device=self._device)
                self._result_queue.put(("load_ok", f"{name} loaded."))
            except Exception as e:
                errors.append((name, str(e)))
        if errors:
            self._result_queue.put(("error", f"Errors while loading: {errors}"))

    @log_decorator
    def run_selected_model(self):
        """Run the model that is currently selected in the dropdown."""
        name = self.model_var.get()
        input_payload = self._gather_input_for_model(name)
        if input_payload is None:
            return
        self._append_output(f"Running {name} ...")
        t = threading.Thread(target=self._run_model_thread, args=(name, input_payload), daemon=True)
        t.start()

    @log_decorator
    def run_alternate_model(self):
        """Run the other model (demonstrates chaining / polymorphism).
        If Text->Image is selected, alt runs ImageClass on generated image, etc.
        """
        selected = self.model_var.get()
        alt_name = [n for n in self._models.keys() if n != selected][0]
        # if we have an on-disk generated image, try classify it
        if selected == "Text-to-Image":
            # run text->image first, then classify
            prompt = self.input_text.get("1.0", tk.END).strip()
            if not prompt:
                messagebox.showwarning("Input required", "Enter a prompt to generate an image for chaining.")
                return
            self._append_output("Running Text->Image and then Image Classification on the generated image ...")
            t = threading.Thread(target=self._chain_text_to_image_then_classify, args=(prompt,), daemon=True)
            t.start()
        else:
            # run the alternate directly on selected input
            input_payload = self._gather_input_for_model(alt_name)
            if input_payload is None:
                return
            self._append_output(f"Running alternate model: {alt_name} ...")
            t = threading.Thread(target=self._run_model_thread, args=(alt_name, input_payload), daemon=True)
            t.start()

    def _chain_text_to_image_then_classify(self, prompt):
        try:
            # ensure both models loaded
            t2i = self._models["Text-to-Image"]
            cls = self._models["Image Classification"]
            if not t2i._is_loaded:
                t2i.load(device=self._device)
            res = t2i.run(prompt)
            # res contains path and pil_image
            img_path = res.get("path")
            # load classification model
            if not cls._is_loaded:
                cls.load()
            cls_res = cls.run(img_path)
            self._result_queue.put(("chain", {"image_path": img_path, "classifications": cls_res}))
        except Exception as e:
            self._result_queue.put(("error", f"Chain error: {e}"))

    def _gather_input_for_model(self, model_name):
        mode = self.input_mode.get()
        if model_name == "Text-to-Image":
            if mode != "Text":
                # if image mode, try to get text from text box anyway
                self._append_output("Text-to-Image expects a text prompt. Using text input box.")
            prompt = self.input_text.get("1.0", tk.END).strip()
            if not prompt:
                messagebox.showwarning("No prompt", "Please enter a text prompt for Text-to-Image.")
                return None
            return prompt
        elif model_name == "Image Classification":
            # prefer a selected image path, else check if user pasted a path in input text
            img_path = getattr(self, "_selected_input_path", None)
            if not img_path:
                text_val = self.input_text.get("1.0", tk.END).strip()
                if os.path.exists(text_val):
                    img_path = text_val
            if not img_path:
                messagebox.showwarning("No image", "Please select an image for classification (Browse -> Image).")
                return None
            return img_path
        else:
            messagebox.showerror("Unsupported", f"Unsupported model: {model_name}")
            return None

    # worker thread executes model.run and posts results to the queue
    def _run_model_thread(self, model_name, input_payload):
        try:
            model = self._models[model_name]
            if not model._is_loaded:
                model.load(device=self._device)
            res = model.run(input_payload)
            self._result_queue.put(("run_result", {"model": model_name, "result": res}))
        except Exception as e:
            self._result_queue.put(("error", f"Error running {model_name}: {e}"))

    # poll queue frequently to update UI from main thread
    def _poll_results(self):
        try:
            while True:
                kind, payload = self._result_queue.get_nowait()
                if kind == "load_ok":
                    self._append_output(payload)
                    self._refresh_model_info()
                elif kind == "run_result":
                    self._handle_model_result(payload)
                elif kind == "chain":
                    self._handle_chain_result(payload)
                elif kind == "error":
                    self._append_output(f"[ERROR] {payload}")
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        # schedule next poll
        self.after(200, self._poll_results)

    # result handlers
    def _handle_model_result(self, payload):
        model_name = payload.get("model")
        res = payload.get("result")
        if res["type"] == "image":
            path = res["path"]
            self._append_output(f"{model_name} generated image: {path}")
            self._display_image(path)
        elif res["type"] == "classifications":
            lines = [f"{model_name} results:"]
            for r in res["results"]:
                lines.append(f"- {r.get('label')}: {r.get('score'):.3f}")
            self._append_output("\n".join(lines))
        else:
            self._append_output(f"{model_name} returned: {res}")

    def _handle_chain_result(self, payload):
        img_path = payload["image_path"]
        cls_res = payload["classifications"]
        self._append_output(f"Generated image saved at: {img_path}")
        self._display_image(img_path)
        # show classification results
        lines = ["Chain classification results:"]
        for r in cls_res:
            lines.append(f"- {r.get('label')}: {r.get('score'):.3f}")
        self._append_output("\n".join(lines))

    # -------------------------
    # UI update helpers
    # -------------------------
    def _append_output(self, text):
        self.output_text.insert(tk.END, text + "\n\n")
        self.output_text.see(tk.END)

    def _display_image(self, path):
        try:
            img = Image.open(path)
            thumb = make_thumbnail(img, size=(420, 320))
            photo = ImageTk.PhotoImage(thumb)
            self._last_photoimage = photo  # keep reference
            self.output_image_label.config(image=photo, text="")
        except Exception as e:
            self._append_output(f"Could not display image: {e}")

    # refresh bottom-left model info text
    def _refresh_model_info(self):
        name = self.model_var.get()
        model = self._models[name]
        info = model.get_info()
        text = (
            f"Model Name: {info['name']}\n"
            f"Category: {info['category']}\n"
            f"Description: {info['description']}\n"
            f"Loaded: {info['loaded']}\n"
        )
        self.model_info_text.config(state="normal")
        self.model_info_text.delete("1.0", tk.END)
        self.model_info_text.insert("1.0", text)
        self.model_info_text.config(state="disabled")

    # refresh bottom-right OOP explanation text (calls overridden method)
    def _refresh_oop_explanation(self):
        explanation = self.explanation()  # overridden method from OOPConcepts
        self.oop_explanation_text.config(state="normal")
        self.oop_explanation_text.delete("1.0", tk.END)
        self.oop_explanation_text.insert("1.0", explanation)
        self.oop_explanation_text.config(state="disabled")
