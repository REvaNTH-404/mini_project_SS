import unittest
from src.signals import SineSignal, SquareSignal, SawtoothSignal

class TestSignals(unittest.TestCase):

    def test_sine_signal(self):
        signal = SineSignal(amplitude=1, frequency=1, phase=0)
        time_range = [0, 1, 0.01]  # start, end, step
        generated_signal = signal.generate(time_range)
        expected_length = int((time_range[1] - time_range[0]) / time_range[2]) + 1
        self.assertEqual(len(generated_signal), expected_length)

    def test_square_signal(self):
        signal = SquareSignal(amplitude=1, frequency=1)
        time_range = [0, 1, 0.01]
        generated_signal = signal.generate(time_range)
        expected_length = int((time_range[1] - time_range[0]) / time_range[2]) + 1
        self.assertEqual(len(generated_signal), expected_length)

    def test_sawtooth_signal(self):
        signal = SawtoothSignal(amplitude=1, frequency=1)
        time_range = [0, 1, 0.01]
        generated_signal = signal.generate(time_range)
        expected_length = int((time_range[1] - time_range[0]) / time_range[2]) + 1
        self.assertEqual(len(generated_signal), expected_length)

if __name__ == '__main__':
    unittest.main()