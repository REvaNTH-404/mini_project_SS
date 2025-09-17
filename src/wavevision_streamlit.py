import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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

# --- F1 Racing / Speed Theme Colors ---
F1_BG = "#181818"
F1_PANEL = "#232323"
F1_ACCENT = "#e10600"
F1_LABEL = "#f4f4f4"
F1_ENTRY_BG = "#282828"
F1_SEPARATOR = F1_ACCENT  # Add this line after your other F1 color variables

st.set_page_config(page_title="WaveVision", layout="wide")

st.markdown(
    f"""
    <div style="background-color:{F1_BG};padding:0;margin:0">
        <h1 style="color:{F1_ACCENT};text-align:center;font-family:'Arial Black',Impact,sans-serif;font-size:3.5rem;">
            🏎️ WAVE VISION 🏎️
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        f"<div style='color:{F1_ACCENT};font-size:1.5rem;font-family:Arial Black;'>Signal 1</div>",
        unsafe_allow_html=True
    )
    sig_type = st.selectbox("Type", ["Sine", "Step", "Ramp", "Impulse", "Sawtooth"])
    amp = st.number_input("Amplitude 1", value=1.0, step=1.0, format="%.2f")
    freq = st.number_input("Frequency 1", value=1.0, step=1.0, format="%.2f")

    st.markdown(
        f"<div style='color:{F1_ACCENT};font-size:1.5rem;font-family:Arial Black;'>Discrete Signal</div>",
        unsafe_allow_html=True
    )
    is_discrete = st.checkbox("Discrete", value=False)
    num_samples = st.number_input("Samples", min_value=2, max_value=1000, value=20, step=1)

    st.markdown(
        f"<div style='color:{F1_ACCENT};font-size:1.5rem;font-family:Arial Black;'>Operations</div>",
        unsafe_allow_html=True
    )
    shift_param = st.slider("Time Shifting", -2.0, 2.0, 0.0, 0.01)
    scaling_param = st.slider("Time Scaling", 0.1, 3.0, 1.0, 0.01)
    amplitude_scaling_param = st.slider("Amplitude Scaling", 0.1, 3.0, 1.0, 0.01)

    st.markdown(
        f"<div style='color:{F1_ACCENT};font-size:1.5rem;font-family:Arial Black;'>Menu</div>",
        unsafe_allow_html=True
    )
    menu_ops = ["None", "Signal Addition", "Signal Multiplication", "Time Reversal"]
    menu_operation = st.selectbox("Operation", menu_ops)

    # Signal 2 controls
    if menu_operation in ["Signal Addition", "Signal Multiplication"]:
        st.markdown(
            f"<div style='color:{F1_ACCENT};font-size:1.2rem;font-family:Arial Black;'>Signal 2</div>",
            unsafe_allow_html=True
        )
        sig2_type = st.selectbox("Type 2", ["Sine", "Step", "Ramp", "Impulse", "Sawtooth"])
        amp2 = st.number_input("Amplitude 2", value=1.0, step=1.0, format="%.2f")
        freq2 = st.number_input("Frequency 2", value=1.0, step=1.0, format="%.2f")
        shift_param2 = st.slider("Time Shift 2", -2.0, 2.0, 0.0, 0.01)
        scaling_param2 = st.slider("Time Scaling 2", 0.1, 3.0, 1.0, 0.01)
        amplitude_scaling_param2 = st.slider("Amplitude Scaling 2", 0.1, 3.0, 1.0, 0.01)
    else:
        sig2_type = None
        amp2 = None
        freq2 = None
        shift_param2 = None
        scaling_param2 = None
        amplitude_scaling_param2 = None

    if menu_operation == "Time Reversal":
        reversal_target = st.selectbox("Reverse Target", ["Original", "Processed"])
    else:
        reversal_target = None

# --- Signal Generation ---
t_input = np.linspace(0, 1, num_samples if is_discrete else 500)
def get_signal(t, amp, freq, sig_type):
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

s_input = get_signal(t_input, amp, freq, sig_type)
t_processed, s_processed = t_input, s_input
t_processed, s_processed = time_shifting(t_processed, s_processed, shift_param)
t_processed, s_processed = time_scaling(t_processed, s_processed, scaling_param)
t_processed, s_processed = t_processed, amplitude_scaling(s_processed, amplitude_scaling_param)

# --- Signal 2 (if needed) ---
if menu_operation in ["Signal Addition", "Signal Multiplication"]:
    s2 = get_signal(t_input, amp2, freq2, sig2_type)
    t2, s2 = time_shifting(t_input, s2, shift_param2)
    t2, s2 = time_scaling(t2, s2, scaling_param2)
    t2, s2 = t2, amplitude_scaling(s2, amplitude_scaling_param2)
else:
    t2, s2 = None, None

# --- Operations ---
t_sum, s_sum = None, None
t_mul, s_mul = None, None
t_rev, s_rev = None, None
if menu_operation == "Signal Addition":
    t_sum, s_sum = t2, signal_addition(s_processed, s2)
elif menu_operation == "Signal Multiplication":
    t_mul, s_mul = t2, signal_multiplication(s_processed, s2)
elif menu_operation == "Time Reversal":
    if reversal_target == "Original":
        t_rev, s_rev = time_reversal(t_input, s_input)
    else:
        t_rev, s_rev = time_reversal(t_processed, s_processed)

# --- Plotting ---
fig, ax = plt.subplots(facecolor=F1_BG)
ax.set_facecolor(F1_BG)
color_original = "#e10600"
color_processed = "#00e1d2"
color_signal2 = "#f7c873"
color_sum = "#6a7fdb"
color_product = "#ff7f50"
color_reversed = "#ff00ff"

ax.axhline(0, color=F1_SEPARATOR, linewidth=1)
ax.axvline(0, color=F1_SEPARATOR, linewidth=1)

if is_discrete:
    ax.stem(t_input, s_input, linefmt=color_original, markerfmt="o", basefmt=" ", label="Original")
    ax.stem(t_processed, s_processed, linefmt=color_processed, markerfmt="o", basefmt=" ", label="Processed")
else:
    ax.plot(t_input, s_input, label="Original", color=color_original, linestyle="--", linewidth=2)
    ax.plot(t_processed, s_processed, label="Processed", color=color_processed, linewidth=2)

if menu_operation == "Signal Addition" and t_sum is not None and s_sum is not None:
    if is_discrete:
        ax.stem(t2, s2, linefmt=color_signal2, markerfmt="o", basefmt=" ", label="Signal 2")
        ax.stem(t_sum, s_sum, linefmt=color_sum, markerfmt="o", basefmt=" ", label="Sum")
    else:
        ax.plot(t2, s2, label="Signal 2", color=color_signal2, linestyle="--", linewidth=2)
        ax.plot(t_sum, s_sum, label="Sum", color=color_sum, linewidth=2)
elif menu_operation == "Signal Multiplication" and t_mul is not None and s_mul is not None:
    if is_discrete:
        ax.stem(t2, s2, linefmt=color_signal2, markerfmt="o", basefmt=" ", label="Signal 2")
        ax.stem(t_mul, s_mul, linefmt=color_product, markerfmt="o", basefmt=" ", label="Product")
    else:
        ax.plot(t2, s2, label="Signal 2", color=color_signal2, linestyle="--", linewidth=2)
        ax.plot(t_mul, s_mul, label="Product", color=color_product, linewidth=2)
elif menu_operation == "Time Reversal" and t_rev is not None and s_rev is not None:
    if is_discrete:
        ax.stem(t_rev, s_rev, linefmt=color_reversed, markerfmt="o", basefmt=" ", label="Reversed")
    else:
        ax.plot(t_rev, s_rev, label="Reversed", color=color_reversed, linewidth=2)

ax.legend(facecolor=F1_PANEL, labelcolor=F1_LABEL, edgecolor=F1_ACCENT)
ax.grid(True, which='major', linestyle='-', color=F1_SEPARATOR, alpha=0.5)
ax.grid(True, which='minor', linestyle=':', color=F1_SEPARATOR, alpha=0.3)
ax.minorticks_on()
ax.set_xlabel("Time (t)", color=F1_LABEL)
ax.set_ylabel("Amplitude", color=F1_LABEL)
ax.set_xlim(-2, 2)
ax.tick_params(colors=F1_LABEL)
fig.tight_layout()

# --- Interactive plot controls ---
st.markdown("### Plot Controls")
plot_col1, plot_col2, plot_col3 = st.columns(3)
with plot_col1:
    zoom_in = st.button("Zoom In")
with plot_col2:
    zoom_out = st.button("Zoom Out")
with plot_col3:
    reset_zoom = st.button("Reset Zoom")
reset_plot = st.button("Reset Plot", type="primary")

# --- Implement zoom and reset logic using session state ---
if "xlim" not in st.session_state:
    st.session_state.xlim = [-2, 2]
if "ylim" not in st.session_state:
    st.session_state.ylim = [ax.get_ylim()[0], ax.get_ylim()[1]]

if zoom_in:
    xmid = sum(st.session_state.xlim) / 2
    xspan = (st.session_state.xlim[1] - st.session_state.xlim[0]) / 4
    st.session_state.xlim = [xmid - xspan, xmid + xspan]
    ymid = sum(st.session_state.ylim) / 2
    yspan = (st.session_state.ylim[1] - st.session_state.ylim[0]) / 4
    st.session_state.ylim = [ymid - yspan, ymid + yspan]
if zoom_out:
    xmid = sum(st.session_state.xlim) / 2
    xspan = (st.session_state.xlim[1] - st.session_state.xlim[0])
    st.session_state.xlim = [xmid - xspan, xmid + xspan]
    ymid = sum(st.session_state.ylim) / 2
    yspan = (st.session_state.ylim[1] - st.session_state.ylim[0])
    st.session_state.ylim = [ymid - yspan, ymid + yspan]
if reset_zoom:
    st.session_state.xlim = [-2, 2]
    st.session_state.ylim = [ax.get_ylim()[0], ax.get_ylim()[1]]
if reset_plot:
    st.experimental_rerun()

# --- Re-plot with updated limits ---
ax.set_xlim(st.session_state.xlim)
ax.set_ylim(st.session_state.ylim)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# --- Parameter information below the plot ---
param_text = (
    f"**Signal 1**  \n"
    f"Time Shift: `{shift_param:.2f}`  \n"
    f"Amplitude Scaling: `{amplitude_scaling_param:.2f}`  \n"
    f"Time Scaling: `{scaling_param:.2f}`"
)
if menu_operation in ["Signal Addition", "Signal Multiplication"]:
    param_text += (
        f"\n\n**Signal 2**  \n"
        f"Time Shift: `{shift_param2:.2f}`  \n"
        f"Amplitude Scaling: `{amplitude_scaling_param2:.2f}`  \n"
        f"Time Scaling: `{scaling_param2:.2f}`"
    )
st.markdown(param_text)

# --- Only keep Reset Plot button ---
if st.button("Reset Plot", type="primary"):
    st.session_state.xlim = [-2, 2]
    st.session_state.ylim = [ax.get_ylim()[0], ax.get_ylim()[1]]
    st.experimental_rerun()