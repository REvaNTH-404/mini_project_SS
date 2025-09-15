def plot_signal(time, signal, title="Signal", xlabel="Time", ylabel="Amplitude"):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def validate_signal_input(signal):
    if not isinstance(signal, (list, np.ndarray)):
        raise ValueError("Signal must be a list or a numpy array.")
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")
    return True

def generate_time_vector(start, end, step):
    import numpy as np
    return np.arange(start, end, step)