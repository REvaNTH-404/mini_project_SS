def time_scaling(t, signal, scaling_factor):
    return t * scaling_factor, signal

def amplitude_scaling(signal, scaling_factor):
    return [x * scaling_factor for x in signal]

def time_shifting(t, signal, shift_amount):
    return t + shift_amount, signal

def time_reversal(t, signal):
    return -t, signal

def signal_addition(signal1, signal2):
    if len(signal1) != len(signal2):
        raise ValueError("Signals must be of the same length for addition.")
    return [a + b for a, b in zip(signal1, signal2)]

def signal_multiplication(signal1, signal2):
    if len(signal1) != len(signal2):
        raise ValueError("Signals must be of the same length for multiplication.")
    return [a * b for a, b in zip(signal1, signal2)]