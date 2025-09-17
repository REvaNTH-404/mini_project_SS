import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from operations import (
    time_scaling,
    amplitude_scaling,
    time_shifting,
    time_reversal,
    signal_addition,
    signal_multiplication
)

def generate_base_signal(t, amp=1, freq=1):
    return amp * np.sin(2 * np.pi * freq * t)

# --- F1 Racing / Speed Theme ---
F1_BG = "#181818"              # Main background (dark asphalt)
F1_PANEL = "#232323"           # Panel background (carbon fiber dark)
F1_ACCENT = "#e10600"          # Red accent (Ferrari/F1 red)
F1_BUTTON = "#232323"          # Button background (carbon fiber dark)
F1_BUTTON_TEXT = "#e10600"     # Button text (F1 red)
F1_LABEL = "#f4f4f4"           # Label text (light gray)
F1_ENTRY_BG = "#282828"        # Entry background (dark gray)
F1_ENTRY_TEXT = "#e10600"      # Entry text (F1 red)
F1_SEPARATOR = "#e10600"       # Separator (F1 red)

class RoundedEntry(ttk.Entry):
    def __init__(self, master=None, **kwargs):
        style_name = kwargs.pop("style", "F1.TEntry")
        super().__init__(master, style=style_name, **kwargs)

def setup_styles():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("F1.TEntry",
        relief="flat",
        borderwidth=8,
        padding=8,
        font=("Segoe UI", 11),
        foreground=F1_ENTRY_TEXT,
        fieldbackground=F1_ENTRY_BG
    )
    style.map("F1.TEntry",
        fieldbackground=[("active", "#232323"), ("!active", F1_ENTRY_BG)],
        bordercolor=[("focus", F1_ACCENT), ("!focus", "#232323")]
    )
    # REDUCE BOLDNESS: Use "normal" instead of "bold" for panel texts
    style.configure("TLabel", background=F1_PANEL, foreground=F1_LABEL, font=("Segoe UI", 12, "normal"))
    style.configure("TButton", background=F1_BUTTON, foreground=F1_BUTTON_TEXT, font=("Segoe UI", 11, "normal"), borderwidth=0, padding=8)
    style.map("TButton", background=[("active", F1_ACCENT)])
    style.configure("TEntry", fieldbackground=F1_ENTRY_BG, foreground=F1_ENTRY_TEXT, font=("Segoe UI", 11), borderwidth=0, padding=8)
    style.configure("TCheckbutton", background=F1_PANEL, foreground=F1_LABEL, font=("Segoe UI", 11, "normal"))
    style.configure("TMenubutton", background=F1_ENTRY_BG, foreground=F1_LABEL, font=("Segoe UI", 11, "normal"))

class WaveVisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WaveVision")
        self.geometry("1200x700")
        self.configure(bg=F1_BG)
        self.resizable(True, True)
        setup_styles()

        # --- Define variables FIRST ---
        self.signal_type = tk.StringVar(value="Sine")
        self.signal2_type = tk.StringVar(value="Sine")
        self.amplitude = tk.DoubleVar(value=1.0)
        self.frequency = tk.DoubleVar(value=1.0)
        self.amplitude2 = tk.DoubleVar(value=1.0)
        self.frequency2 = tk.DoubleVar(value=1.0)
        self.is_discrete = tk.BooleanVar(value=False)
        self.num_samples = tk.IntVar(value=20)

        self.shift_param = tk.DoubleVar(value=0.0)
        self.scaling_param = tk.DoubleVar(value=1.0)
        self.amplitude_scaling_param = tk.DoubleVar(value=1.0)

        # New variables for signal 2 transformations
        self.shift_param2 = tk.DoubleVar(value=0.0)
        self.scaling_param2 = tk.DoubleVar(value=1.0)
        self.amplitude_scaling_param2 = tk.DoubleVar(value=1.0)

        self.menu_operation = tk.StringVar(value="None")
        self.reversal_target = tk.StringVar(value="Processed")

        # --- Controls panel (left) ---
        control_frame = tk.Frame(self, width=370, bg=F1_PANEL, highlightbackground=F1_ACCENT, highlightthickness=4)
        control_frame.pack(side="left", fill="y", padx=0, pady=0)
        control_frame.pack_propagate(False)

        # F1-themed header with racing flag emoji
        header_frame = tk.Frame(control_frame, bg=F1_PANEL)
        header_frame.pack(fill="x", pady=(0, 10))
        header_label = tk.Label(
            header_frame,
            text="🏁 WAVE VISION 🏁",
            font=("Arial Black", 26, "bold"),
            fg=F1_ACCENT,
            bg=F1_PANEL,
            pady=16
        )
        header_label.pack(fill="x")

        sep = tk.Frame(control_frame, bg=F1_ACCENT, height=3)
        sep.pack(fill="x", padx=18, pady=(0, 12))

        # --- Section: Signal 1 ---
        section1 = tk.LabelFrame(control_frame, text="Signal 1", bg=F1_PANEL, fg=F1_ACCENT, font=("Segoe UI", 13, "bold"), bd=2, relief="ridge", labelanchor="n")
        section1.pack(fill="x", padx=18, pady=(0, 12))

        ttk.Label(section1, text="Type").pack(pady=(10, 0))
        signal_types = ["Sine", "Step", "Ramp", "Impulse", "Sawtooth"]
        signal1_menu = tk.OptionMenu(
            section1, self.signal_type, self.signal_type.get(), *signal_types, command=lambda _: self.plot()
        )
        signal1_menu.config(
            bg=F1_ENTRY_BG,
            fg=F1_LABEL,
            activebackground="#353535",
            activeforeground=F1_ACCENT,
            highlightthickness=0,
            bd=0
        )
        signal1_menu["menu"].config(
            bg=F1_ENTRY_BG,
            fg=F1_LABEL,
            activebackground="#353535",
            activeforeground=F1_ACCENT
        )
        signal1_menu.pack(fill="x", padx=10)

        ttk.Label(section1, text="Amplitude").pack(pady=(10, 0))
        amp1_frame = tk.Frame(section1, bg=F1_PANEL)
        amp1_frame.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(amp1_frame, text="-", width=2, command=lambda: self._inc_dec(self.amplitude, -1)).pack(side="left")
        RoundedEntry(amp1_frame, textvariable=self.amplitude).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(amp1_frame, text="+", width=2, command=lambda: self._inc_dec(self.amplitude, 1)).pack(side="left")
        self.amplitude.trace_add("write", lambda *args: self.plot())

        ttk.Label(section1, text="Frequency").pack(pady=(10, 0))
        freq1_frame = tk.Frame(section1, bg=F1_PANEL)
        freq1_frame.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(freq1_frame, text="-", width=2, command=lambda: self._inc_dec(self.frequency, -1)).pack(side="left")
        RoundedEntry(freq1_frame, textvariable=self.frequency).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(freq1_frame, text="+", width=2, command=lambda: self._inc_dec(self.frequency, 1)).pack(side="left")
        self.frequency.trace_add("write", lambda *args: self.plot())

        # --- Section: Discrete Controls ---
        discrete_section = tk.LabelFrame(control_frame, text="Discrete Signal", bg=F1_PANEL, fg=F1_ACCENT, font=("Segoe UI", 13, "bold"), bd=2, relief="ridge", labelanchor="n")
        discrete_section.pack(fill="x", padx=18, pady=(0, 12))

        ttk.Checkbutton(discrete_section, text="Discrete", variable=self.is_discrete, command=self._toggle_samples_entry).pack(side="left", padx=(10, 10), pady=8)
        self.samples_label = ttk.Label(discrete_section, text="Samples")
        self.samples_entry = RoundedEntry(discrete_section, textvariable=self.num_samples)
        self.num_samples.trace_add("write", lambda *args: self.plot())

        # --- Section: Operations ---
        section2 = tk.LabelFrame(
            control_frame,
            text="Operations",
            bg=F1_PANEL,
            fg=F1_ACCENT,
            font=("Segoe UI", 13, "bold"),
            bd=2,
            relief="ridge",
            labelanchor="n"
        )
        section2.pack(fill="x", padx=18, pady=(0, 12))

        ttk.Label(section2, text="Time Shifting").pack(pady=(10, 0))
        ttk.Scale(section2, from_=-2.0, to=2.0, variable=self.shift_param, command=lambda _: self.plot()).pack(fill="x", padx=10)
        ttk.Label(section2, text="Time Scaling").pack(pady=(10, 0))
        ttk.Scale(section2, from_=0.1, to=3.0, variable=self.scaling_param, command=lambda _: self.plot()).pack(fill="x", padx=10)
        ttk.Label(section2, text="Amplitude Scaling").pack(pady=(10, 0))
        ttk.Scale(section2, from_=0.1, to=3.0, variable=self.amplitude_scaling_param, command=lambda _: self.plot()).pack(fill="x", padx=10)

        # --- Section: Menu Operations ---
        section3 = tk.LabelFrame(control_frame, text="Menu", bg=F1_PANEL, fg=F1_ACCENT, font=("Segoe UI", 13, "bold"), bd=2, relief="ridge", labelanchor="n")
        section3.pack(fill="x", padx=18, pady=(0, 12))

        ttk.Label(section3, text="Operation").pack(pady=(10, 0))
        menu_ops = ["None", "Signal Addition", "Signal Multiplication", "Time Reversal"]
        menu_frame = tk.Frame(section3, bg=F1_PANEL)
        menu_frame.pack(fill="x", padx=10)
        menu_option = tk.OptionMenu(
            menu_frame, self.menu_operation, self.menu_operation.get(), *menu_ops, command=lambda _: self.plot()
        )
        menu_option.config(
            bg=F1_ENTRY_BG,
            fg=F1_LABEL,
            activebackground="#353535",
            activeforeground=F1_ACCENT,
            highlightthickness=0,
            bd=0
        )
        menu_option["menu"].config(
            bg=F1_ENTRY_BG,
            fg=F1_LABEL,
            activebackground="#353535",
            activeforeground=F1_ACCENT
        )
        menu_option.pack(fill="x")

        # Signal 2 controls (for addition/multiplication)
        self.signal2_controls = tk.Frame(section3, bg=F1_PANEL)
        ttk.Label(self.signal2_controls, text="Signal 2 Type").pack(pady=(10, 0))
        ttk.OptionMenu(self.signal2_controls, self.signal2_type, self.signal2_type.get(), *signal_types, command=lambda _: self.plot()).pack(fill="x", padx=10)

        # --- Amplitude 2 with +/- buttons ---
        ttk.Label(self.signal2_controls, text="Amplitude 2").pack(pady=(10, 0))
        amp2_frame = tk.Frame(self.signal2_controls, bg=F1_PANEL)
        amp2_frame.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(amp2_frame, text="-", width=2, command=lambda: self._inc_dec(self.amplitude2, -1)).pack(side="left")
        RoundedEntry(amp2_frame, textvariable=self.amplitude2).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(amp2_frame, text="+", width=2, command=lambda: self._inc_dec(self.amplitude2, 1)).pack(side="left")
        self.amplitude2.trace_add("write", lambda *args: self.plot())

        # --- Frequency 2 with +/- buttons ---
        ttk.Label(self.signal2_controls, text="Frequency 2").pack(pady=(10, 0))
        freq2_frame = tk.Frame(self.signal2_controls, bg=F1_PANEL)
        freq2_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(freq2_frame, text="-", width=2, command=lambda: self._inc_dec(self.frequency2, -1)).pack(side="left")
        RoundedEntry(freq2_frame, textvariable=self.frequency2).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(freq2_frame, text="+", width=2, command=lambda: self._inc_dec(self.frequency2, 1)).pack(side="left")
        self.frequency2.trace_add("write", lambda *args: self.plot())

        # --- Signal 2 Transformations ---
        ttk.Label(self.signal2_controls, text="Time Shift 2").pack(pady=(10, 0))
        ttk.Scale(self.signal2_controls, from_=-2.0, to=2.0, variable=self.shift_param2, command=lambda _: self.plot()).pack(fill="x", padx=10)
        ttk.Label(self.signal2_controls, text="Time Scaling 2").pack(pady=(10, 0))
        ttk.Scale(self.signal2_controls, from_=0.1, to=3.0, variable=self.scaling_param2, command=lambda _: self.plot()).pack(fill="x", padx=10)
        ttk.Label(self.signal2_controls, text="Amplitude Scaling 2").pack(pady=(10, 0))
        ttk.Scale(self.signal2_controls, from_=0.1, to=3.0, variable=self.amplitude_scaling_param2, command=lambda _: self.plot()).pack(fill="x", padx=10)

        # Time reversal controls
        self.reversal_controls = tk.Frame(section3, bg=F1_PANEL)
        ttk.Label(self.reversal_controls, text="Reverse Target").pack(pady=(10, 0))
        ttk.OptionMenu(self.reversal_controls, self.reversal_target, self.reversal_target.get(), "Original", "Processed", command=lambda _: self.plot()).pack(fill="x", padx=10)

        # --- Plot area (right) ---
        plot_frame = tk.Frame(self, bg=F1_BG)
        plot_frame.pack(side="right", fill="both", expand=True)

        # --- F1 Logo ---
        logo_frame = tk.Frame(plot_frame, bg=F1_BG)
        logo_frame.pack(side="top", fill="x", pady=(30, 0))
        logo_label = tk.Label(
            logo_frame,
            text="🏎️ WAVE VISION",
            font=("Arial Black", 54, "bold"),
            fg=F1_ACCENT,
            bg=F1_BG
        )
        logo_label.pack(pady=(0, 10))

        self.fig, self.ax = plt.subplots(facecolor=F1_BG)
        self.ax.set_facecolor(F1_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Coordinate display label ---
        self.coord_label = tk.Label(plot_frame, text="Time: ---, Amplitude: ---", bg=F1_BG, fg=F1_ACCENT, font=("Segoe UI", 11))
        self.coord_label.pack(anchor="se", padx=10, pady=2, side="bottom")

        # Connect the mpl_connect event for mouse motion
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_hover)

        zoom_frame = tk.Frame(plot_frame, bg=F1_BG)
        zoom_frame.pack(side="bottom", anchor="w", pady=10)
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side="left", padx=5)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side="left", padx=5)
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side="left", padx=5)
        ttk.Button(zoom_frame, text="Reset Plot", command=self.reset_plot).pack(side="left", padx=5)

        widget = self.canvas.get_tk_widget()
        widget.bind('<MouseWheel>', self._on_mousewheel_zoom)
        widget.bind('<Button-4>', self._on_mousewheel_zoom)
        widget.bind('<Button-5>', self._on_mousewheel_zoom)
        widget.bind('<ButtonPress-1>', self._on_pan_start)
        widget.bind('<B1-Motion>', self._on_pan_move)
        widget.bind('<ButtonRelease-1>', self._on_pan_end)

        self._default_xlim = None
        self._default_ylim = None
        self._dragging = False
        self._drag_start = None
        self._orig_xlim = None
        self._orig_ylim = None

        # --- Now call toggle_samples_entry ---
        self._toggle_samples_entry()

        self.plot()

    def plot(self):
        is_discrete = self.is_discrete.get()
        num_samples = self.num_samples.get()
        t_input = np.linspace(0, 1, num_samples if is_discrete else 500)  # <-- FIXED HERE
        amp = self.amplitude.get()
        freq = self.frequency.get()
        sig_type = self.signal_type.get()
        s_input = self._get_signal(t_input, amp, freq, sig_type)

        t_processed, s_processed = t_input, s_input
        t_processed, s_processed = time_shifting(t_processed, s_processed, self.shift_param.get())
        t_processed, s_processed = time_scaling(t_processed, s_processed, self.scaling_param.get())
        t_processed, s_processed = t_processed, amplitude_scaling(s_processed, self.amplitude_scaling_param.get())

        if self.menu_operation.get() in ["Signal Addition", "Signal Multiplication"]:
            self.signal2_controls.pack(fill="x", pady=5)
            self.reversal_controls.pack_forget()
        elif self.menu_operation.get() == "Time Reversal":
            self.signal2_controls.pack_forget()
            self.reversal_controls.pack(fill="x", pady=5)
        else:
            self.signal2_controls.pack_forget()
            self.reversal_controls.pack_forget()

        t_sum, s_sum = None, None
        t_mul, s_mul = None, None
        t_rev, s_rev = None, None
        if self.menu_operation.get() == "Signal Addition":
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            # Apply signal 2 transformations
            t2, s2 = time_shifting(t_input, s2, self.shift_param2.get())
            t2, s2 = time_scaling(t2, s2, self.scaling_param2.get())
            t2, s2 = t2, amplitude_scaling(s2, self.amplitude_scaling_param2.get())
            t_sum, s_sum = t2, signal_addition(s_processed, s2)
        elif self.menu_operation.get() == "Signal Multiplication":
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            # Apply signal 2 transformations
            t2, s2 = time_shifting(t_input, s2, self.shift_param2.get())
            t2, s2 = time_scaling(t2, s2, self.scaling_param2.get())
            t2, s2 = t2, amplitude_scaling(s2, self.amplitude_scaling_param2.get())
            t_mul, s_mul = t2, signal_multiplication(s_processed, s2)
        elif self.menu_operation.get() == "Time Reversal":
            target = self.reversal_target.get()
            if target == "Original":
                t_rev, s_rev = time_reversal(t_input, s_input)
            else:
                t_rev, s_rev = time_reversal(t_processed, s_processed)

        # --- Distinct colors for each signal ---
        color_original = "#e10600"      # F1 red
        color_processed = "#00e1d2"     # Cyan
        color_signal2 = "#f7c873"       # Yellow
        color_sum = "#6a7fdb"           # Blue
        color_product = "#ff7f50"       # Coral
        color_reversed = "#ff00ff"      # Magenta

        self.ax.clear()
        self.ax.axhline(0, color=F1_SEPARATOR, linewidth=1)
        self.ax.axvline(0, color=F1_SEPARATOR, linewidth=1)

        if is_discrete:
            self.ax.stem(t_input, s_input, linefmt=color_original, markerfmt="o", basefmt=" ", label="Original")
            self.ax.stem(t_processed, s_processed, linefmt=color_processed, markerfmt="o", basefmt=" ", label="Processed")
        else:
            self.ax.plot(t_input, s_input, label="Original", color=color_original, linestyle="--", linewidth=2)
            self.ax.plot(t_processed, s_processed, label="Processed", color=color_processed, linewidth=2)

        if self.menu_operation.get() == "Signal Addition" and t_sum is not None and s_sum is not None:
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            t2, s2 = time_shifting(t_input, s2, self.shift_param2.get())
            t2, s2 = time_scaling(t2, s2, self.scaling_param2.get())
            t2, s2 = t2, amplitude_scaling(s2, self.amplitude_scaling_param2.get())
            if is_discrete:
                self.ax.stem(t2, s2, linefmt=color_signal2, markerfmt="o", basefmt=" ", label="Signal 2")
                self.ax.stem(t_sum, s_sum, linefmt=color_sum, markerfmt="o", basefmt=" ", label="Sum")
            else:
                self.ax.plot(t2, s2, label="Signal 2", color=color_signal2, linestyle="--", linewidth=2)
                self.ax.plot(t_sum, s_sum, label="Sum", color=color_sum, linewidth=2)
        elif self.menu_operation.get() == "Signal Multiplication" and t_mul is not None and s_mul is not None:
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            t2, s2 = time_shifting(t_input, s2, self.shift_param2.get())
            t2, s2 = time_scaling(t2, s2, self.scaling_param2.get())
            t2, s2 = t2, amplitude_scaling(s2, self.amplitude_scaling_param2.get())
            if is_discrete:
                self.ax.stem(t2, s2, linefmt=color_signal2, markerfmt="o", basefmt=" ", label="Signal 2")
                self.ax.stem(t_mul, s_mul, linefmt=color_product, markerfmt="o", basefmt=" ", label="Product")
            else:
                self.ax.plot(t2, s2, label="Signal 2", color=color_signal2, linestyle="--", linewidth=2)
                self.ax.plot(t_mul, s_mul, label="Product", color=color_product, linewidth=2)
        elif self.menu_operation.get() == "Time Reversal" and t_rev is not None and s_rev is not None:
            if is_discrete:
                self.ax.stem(t_rev, s_rev, linefmt=color_reversed, markerfmt="o", basefmt=" ", label="Reversed")
            else:
                self.ax.plot(t_rev, s_rev, label="Reversed", color=color_reversed, linewidth=2)

        # --- Add parameter box in bottom right ---
        shift_val = self.shift_param.get()
        amp_scale_val = self.amplitude_scaling_param.get()
        time_scale_val = self.scaling_param.get()
        param_text = (
            f"Signal 1\n"
            f"Time Shift: {shift_val:.2f}\n"
            f"Amplitude Scaling: {amp_scale_val:.2f}\n"
            f"Time Scaling: {time_scale_val:.2f}"
        )

        # If Signal 2 controls are visible, show their parameters as well
        if self.menu_operation.get() in ["Signal Addition", "Signal Multiplication"]:
            shift_val2 = self.shift_param2.get()
            amp_scale_val2 = self.amplitude_scaling_param2.get()
            time_scale_val2 = self.scaling_param2.get()
            param_text += (
                f"\n\nSignal 2\n"
                f"Time Shift: {shift_val2:.2f}\n"
                f"Amplitude Scaling: {amp_scale_val2:.2f}\n"
                f"Time Scaling: {time_scale_val2:.2f}"
            )

        self.ax.text(
            0.98, 0.02, param_text,
            transform=self.ax.transAxes,
            fontsize=12,
            color="#f4f4f4",
            ha="right", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#232323",
                edgecolor="#e10600",
                alpha=0.85
            )
        )

        self.ax.legend(facecolor=F1_PANEL, labelcolor=F1_LABEL, edgecolor=F1_ACCENT)
        self.ax.grid(True, which='major', linestyle='-', color=F1_SEPARATOR, alpha=0.5)
        self.ax.grid(True, which='minor', linestyle=':', color=F1_SEPARATOR, alpha=0.3)
        self.ax.minorticks_on()
        self.ax.set_xlabel("Time (t)", color=F1_LABEL)
        self.ax.set_ylabel("Amplitude", color=F1_LABEL)
        self.ax.set_xlim(-2, 2)
        self.ax.tick_params(colors=F1_LABEL)
        self.fig.tight_layout()
        self.canvas.draw()

        if self._default_xlim is None or self._default_ylim is None:
            self._default_xlim = self.ax.get_xlim()
            self._default_ylim = self.ax.get_ylim()

    def zoom_in(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        xspan = (xlim[1] - xlim[0]) * 0.5
        yspan = (ylim[1] - ylim[0]) * 0.5
        self.ax.set_xlim(xmid - xspan/2, xmid + xspan/2)
        self.ax.set_ylim(ymid - yspan/2, ymid + yspan/2)
        self.canvas.draw()

    def zoom_out(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        xspan = (xlim[1] - xlim[0]) * 2
        yspan = (ylim[1] - ylim[0]) * 2
        self.ax.set_xlim(xmid - xspan/2, xmid + xspan/2)
        self.ax.set_ylim(ymid - yspan/2, ymid + yspan/2)
        self.canvas.draw()

    def reset_zoom(self):
        if self._default_xlim and self._default_ylim:
            self.ax.set_xlim(self._default_xlim)
            self.ax.set_ylim(self._default_ylim)
            self.canvas.draw()

    def reset_plot(self):
        # Reset all controls to their default values
        self.signal_type.set("Sine")
        self.signal2_type.set("Sine")
        self.amplitude.set(1.0)
        self.frequency.set(1.0)
        self.amplitude2.set(1.0)
        self.frequency2.set(1.0)
        self.is_discrete.set(False)
        self.num_samples.set(20)
        self.shift_param.set(0.0)
        self.scaling_param.set(1.0)
        self.amplitude_scaling_param.set(1.0)
        self.menu_operation.set("None")
        self.reversal_target.set("Processed")
        self._toggle_samples_entry()
        self.plot()  # Ensure plot updates immediately on reset

    def _on_mousewheel_zoom(self, event):
        # Windows: event.delta, Linux: event.num
        if hasattr(event, 'delta') and event.delta:
            direction = 1 if event.delta > 0 else -1
        elif hasattr(event, 'num'):
            direction = 1 if event.num == 4 else -1
        else:
            direction = 0
        base = 1.2
        factor = base if direction > 0 else 1/base
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        xspan = (xlim[1] - xlim[0]) / factor
        yspan = (ylim[1] - ylim[0]) / factor
        self.ax.set_xlim(xmid - xspan/2, xmid + xspan/2)
        self.ax.set_ylim(ymid - yspan/2, ymid + yspan/2)
        self.canvas.draw()

    def _on_pan_start(self, event):
        self._dragging = True
        self._drag_start = (event.x, event.y)
        self._orig_xlim = self.ax.get_xlim()
        self._orig_ylim = self.ax.get_ylim()

    def _on_pan_move(self, event):
        if not self._dragging or self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        xlim = self._orig_xlim
        ylim = self._orig_ylim
        widget = self.canvas.get_tk_widget()
        width = widget.winfo_width()
        height = widget.winfo_height()
        if width == 0 or height == 0:
            return
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        x_shift = -dx * x_range / width
        y_shift = dy * y_range / height
        self.ax.set_xlim(xlim[0] + x_shift, xlim[1] + x_shift)
        self.ax.set_ylim(ylim[0] + y_shift, ylim[1] + y_shift)
        self.canvas.draw()

    def _on_pan_end(self, event):
        self._dragging = False
        self._drag_start = None
        self._orig_xlim = None
        self._orig_ylim = None

    def _get_signal(self, t, amp, freq, sig_type):
        if sig_type == "Sine":
            return generate_base_signal(t, amp=amp, freq=freq)
        elif sig_type == "Step":
            return amp * np.heaviside(t, 1)
        elif sig_type == "Ramp":
            return amp * t
        elif sig_type == "Impulse":
            s = np.zeros_like(t)
            s[len(s)//2] = amp
            return s
        elif sig_type == "Sawtooth":
            return amp * (2 * (freq * t - np.floor(0.5 + freq * t)))
        else:
            return generate_base_signal(t, amp=amp, freq=freq)

    def _toggle_samples_entry(self):
        # Always update plot and show/hide samples entry dynamically
        if self.is_discrete.get():
            self.samples_label.pack()
            self.samples_entry.pack(fill="x")
        else:
            self.samples_label.pack_forget()
            self.samples_entry.pack_forget()
        self.plot()  # Ensure plot updates immediately on toggle

    def _inc_dec(self, var, delta):
        try:
            value = var.get()
            value = float(value)
        except Exception:
            value = 0
        value += delta
        if value < 0:
            value = 0
        var.set(round(value, 2))

    def _on_plot_hover(self, event):
        # Remove previous marker/text if any
        if hasattr(self, "_hover_marker"):
            self._hover_marker.remove()
            del self._hover_marker
        if hasattr(self, "_hover_text"):
            self._hover_text.remove()
            del self._hover_text

        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            # Show values in the label
            self.coord_label.config(
                text=f"Time: {event.xdata:.3f}, Amplitude: {event.ydata:.3f}"
            )
            # Draw a marker and value text at the cursor
            self._hover_marker = self.ax.plot(event.xdata, event.ydata, marker="o", color="#e10600", markersize=8, zorder=10)[0]
            self._hover_text = self.ax.text(
                event.xdata, event.ydata,
                f"({event.xdata:.2f}, {event.ydata:.2f})",
                color="#f4f4f4", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#232323", ec="#e10600", alpha=0.8),
                ha="left", va="bottom", zorder=11
            )
            self.canvas.draw_idle()
        else:
            self.coord_label.config(
                text="Time: ---, Amplitude: ---"
            )
            self.canvas.draw_idle()

if __name__ == "__main__":
    app = WaveVisionApp()
    app.mainloop()