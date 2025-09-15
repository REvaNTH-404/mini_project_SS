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

class WaveVisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WaveVision")
        self.geometry("1000x600")
        self.resizable(True, True)

        # Variables
        self.signal_type = tk.StringVar(value="Sine")
        self.signal2_type = tk.StringVar(value="Sine")
        self.amplitude = tk.DoubleVar(value=1.0)
        self.frequency = tk.DoubleVar(value=1.0)
        self.amplitude2 = tk.DoubleVar(value=1.0)
        self.frequency2 = tk.DoubleVar(value=1.0)
        self.is_discrete = tk.BooleanVar(value=False)
        self.num_samples = tk.IntVar(value=20)

        # Operation sliders
        self.shift_param = tk.DoubleVar(value=0.0)
        self.scaling_param = tk.DoubleVar(value=1.0)
        self.amplitude_scaling_param = tk.DoubleVar(value=1.0)

        # Menu operation variables
        self.menu_operation = tk.StringVar(value="None")
        self.reversal_target = tk.StringVar(value="Processed")

        # Controls
        control_frame = tk.Frame(self)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        ttk.Label(control_frame, text="Time Shifting").pack()
        ttk.Scale(control_frame, from_=-2.0, to=2.0, variable=self.shift_param, command=lambda _: self.plot()).pack(fill="x")
        ttk.Label(control_frame, text="Time Scaling").pack()
        ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.scaling_param, command=lambda _: self.plot()).pack(fill="x")
        ttk.Label(control_frame, text="Amplitude Scaling").pack()
        ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.amplitude_scaling_param, command=lambda _: self.plot()).pack(fill="x")

        ttk.Label(control_frame, text="Signal 1 Type").pack()
        signal_types = ["Sine", "Step", "Ramp", "Impulse", "Sawtooth"]
        ttk.OptionMenu(control_frame, self.signal_type, self.signal_type.get(), *signal_types, command=lambda _: self.plot()).pack(fill="x")
        ttk.Label(control_frame, text="Amplitude 1").pack()
        ttk.Entry(control_frame, textvariable=self.amplitude).pack(fill="x")
        self.amplitude.trace_add("write", lambda *args: self.plot())
        ttk.Label(control_frame, text="Frequency 1").pack()
        ttk.Entry(control_frame, textvariable=self.frequency).pack(fill="x")
        self.frequency.trace_add("write", lambda *args: self.plot())

        # Samples entry (only for discrete)
        self.samples_label = ttk.Label(control_frame, text="Samples")
        self.samples_entry = ttk.Entry(control_frame, textvariable=self.num_samples)
        self.num_samples.trace_add("write", lambda *args: self.plot())

        # Discrete checkbox
        ttk.Checkbutton(control_frame, text="Discrete", variable=self.is_discrete, command=self._toggle_samples_entry).pack()
        self._toggle_samples_entry()  # Initial state

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=8)

        # Menu for additional operations
        ttk.Label(control_frame, text="Menu Operations").pack(pady=(8,0))
        menu_ops = ["None", "Signal Addition", "Signal Multiplication", "Time Reversal"]
        ttk.OptionMenu(control_frame, self.menu_operation, self.menu_operation.get(), *menu_ops, command=lambda _: self.plot()).pack(fill="x")

        # Signal Addition/Multiplication controls
        self.signal2_controls = tk.Frame(control_frame)
        ttk.Label(self.signal2_controls, text="Signal 2 Type").pack()
        ttk.OptionMenu(self.signal2_controls, self.signal2_type, self.signal2_type.get(), *signal_types, command=lambda _: self.plot()).pack(fill="x")
        ttk.Label(self.signal2_controls, text="Amplitude 2").pack()
        ttk.Entry(self.signal2_controls, textvariable=self.amplitude2).pack(fill="x")
        self.amplitude2.trace_add("write", lambda *args: self.plot())
        ttk.Label(self.signal2_controls, text="Frequency 2").pack()
        ttk.Entry(self.signal2_controls, textvariable=self.frequency2).pack(fill="x")
        self.frequency2.trace_add("write", lambda *args: self.plot())

        # Time Reversal controls
        self.reversal_controls = tk.Frame(control_frame)
        ttk.Label(self.reversal_controls, text="Reverse Target").pack()
        ttk.OptionMenu(self.reversal_controls, self.reversal_target, self.reversal_target.get(), "Original", "Processed", command=lambda _: self.plot()).pack(fill="x")

        # Plot area
        plot_frame = tk.Frame(self)
        plot_frame.pack(side="right", fill="both", expand=True)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Zoom controls
        zoom_frame = tk.Frame(plot_frame)
        zoom_frame.pack(side="bottom", anchor="w", pady=5)
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side="left", padx=5)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side="left", padx=5)
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side="left", padx=5)

        # Mouse wheel and drag events for zoom/pan
        widget = self.canvas.get_tk_widget()
        widget.bind('<MouseWheel>', self._on_mousewheel_zoom)  # Windows
        widget.bind('<Button-4>', self._on_mousewheel_zoom)    # Linux scroll up
        widget.bind('<Button-5>', self._on_mousewheel_zoom)    # Linux scroll down
        widget.bind('<ButtonPress-1>', self._on_pan_start)
        widget.bind('<B1-Motion>', self._on_pan_move)
        widget.bind('<ButtonRelease-1>', self._on_pan_end)

        self._default_xlim = None
        self._default_ylim = None
        self._dragging = False
        self._drag_start = None
        self._orig_xlim = None
        self._orig_ylim = None

        self.plot()

    def plot(self):
        is_discrete = self.is_discrete.get()
        num_samples = self.num_samples.get()
        t_input = np.linspace(0, 2, num_samples if is_discrete else 500)
        amp = self.amplitude.get()
        freq = self.frequency.get()
        sig_type = self.signal_type.get()
        s_input = self._get_signal(t_input, amp, freq, sig_type)

        # Apply all three operations in sequence
        t_processed, s_processed = t_input, s_input
        t_processed, s_processed = time_shifting(t_processed, s_processed, self.shift_param.get())
        t_processed, s_processed = time_scaling(t_processed, s_processed, self.scaling_param.get())
        t_processed, s_processed = t_processed, amplitude_scaling(s_processed, self.amplitude_scaling_param.get())

        # Show/hide menu controls
        if self.menu_operation.get() in ["Signal Addition", "Signal Multiplication"]:
            self.signal2_controls.pack(fill="x", pady=5)
            self.reversal_controls.pack_forget()
        elif self.menu_operation.get() == "Time Reversal":
            self.signal2_controls.pack_forget()
            self.reversal_controls.pack(fill="x", pady=5)
        else:
            self.signal2_controls.pack_forget()
            self.reversal_controls.pack_forget()

        # Apply menu operation
        t_sum, s_sum = None, None
        t_mul, s_mul = None, None
        t_rev, s_rev = None, None
        if self.menu_operation.get() == "Signal Addition":
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            t_sum, s_sum = t_input, signal_addition(s_processed, s2)
        elif self.menu_operation.get() == "Signal Multiplication":
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            t_mul, s_mul = t_input, signal_multiplication(s_processed, s2)
        elif self.menu_operation.get() == "Time Reversal":
            target = self.reversal_target.get()
            if target == "Original":
                t_rev, s_rev = time_reversal(t_input, s_input)
            else:
                t_rev, s_rev = time_reversal(t_processed, s_processed)

        self.ax.clear()
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.axvline(0, color='black', linewidth=1)

        # Plot original and processed
        if is_discrete:
            self.ax.stem(t_input, s_input, linefmt="b", markerfmt="bo", basefmt=" ", label="Original")
            self.ax.stem(t_processed, s_processed, linefmt="r", markerfmt="ro", basefmt=" ", label="Processed")
        else:
            self.ax.plot(t_input, s_input, label="Original", color="blue", linestyle="--")
            self.ax.plot(t_processed, s_processed, label="Processed", color="red", linewidth=2)

        # Plot menu operation results
        if self.menu_operation.get() == "Signal Addition" and t_sum is not None and s_sum is not None:
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            if is_discrete:
                self.ax.stem(t_input, s2, linefmt="g", markerfmt="go", basefmt=" ", label="Signal 2")
                self.ax.stem(t_sum, s_sum, linefmt="m", markerfmt="mo", basefmt=" ", label="Sum")
            else:
                self.ax.plot(t_input, s2, label="Signal 2", color="green", linestyle="--")
                self.ax.plot(t_sum, s_sum, label="Sum", color="magenta", linewidth=2)
        elif self.menu_operation.get() == "Signal Multiplication" and t_mul is not None and s_mul is not None:
            amp2 = self.amplitude2.get()
            freq2 = self.frequency2.get()
            sig2_type = self.signal2_type.get()
            s2 = self._get_signal(t_input, amp2, freq2, sig2_type)
            if is_discrete:
                self.ax.stem(t_input, s2, linefmt="g", markerfmt="go", basefmt=" ", label="Signal 2")
                self.ax.stem(t_mul, s_mul, linefmt="y", markerfmt="yo", basefmt=" ", label="Product")
            else:
                self.ax.plot(t_input, s2, label="Signal 2", color="green", linestyle="--")
                self.ax.plot(t_mul, s_mul, label="Product", color="orange", linewidth=2)
        elif self.menu_operation.get() == "Time Reversal" and t_rev is not None and s_rev is not None:
            if is_discrete:
                self.ax.stem(t_rev, s_rev, linefmt="c", markerfmt="co", basefmt=" ", label="Reversed")
            else:
                self.ax.plot(t_rev, s_rev, label="Reversed", color="cyan", linewidth=2)

        self.ax.legend()
        self.ax.grid(True, which='major', linestyle='-', color='gray', alpha=0.5)
        self.ax.grid(True, which='minor', linestyle=':', color='lightgray', alpha=0.3)
        self.ax.minorticks_on()
        self.ax.set_xlabel("Time (t)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(-2, 2)
        self.fig.tight_layout()
        self.canvas.draw()

        # Save default limits for reset
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
        if self.is_discrete.get():
            self.samples_label.pack()
            self.samples_entry.pack(fill="x")
        else:
            self.samples_label.pack_forget()
            self.samples_entry.pack_forget()

if __name__ == "__main__":
    app = WaveVisionApp()
    app.mainloop()