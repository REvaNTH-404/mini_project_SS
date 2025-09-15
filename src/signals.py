import numpy as np
class Signal:
    def __init__(self, amplitude=1, frequency=1, phase=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def generate(self, t):
        raise NotImplementedError("This method should be overridden by subclasses")


class SineSignal(Signal):
    def generate(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)


class SquareSignal(Signal):
    def generate(self, t):
        return self.amplitude * np.sign(np.sin(2 * np.pi * self.frequency * t + self.phase))


class SawtoothSignal(Signal):
    def generate(self, t):
        return self.amplitude * (2 * (t * self.frequency - np.floor(0.5 + t * self.frequency)))


def create_signal(signal_type, amplitude=1, frequency=1, phase=0):
    if signal_type == 'sine':
        return SineSignal(amplitude, frequency, phase)
    elif signal_type == 'square':
        return SquareSignal(amplitude, frequency, phase)
    elif signal_type == 'sawtooth':
        return SawtoothSignal(amplitude, frequency, phase)
    else:
        raise ValueError("Unsupported signal type")