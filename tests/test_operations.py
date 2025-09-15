import unittest
from src.signals import SineSignal, SquareSignal, SawtoothSignal
from src.operations import time_scaling, amplitude_scaling, time_shifting, time_reversal, signal_addition, signal_multiplication

class TestSignalOperations(unittest.TestCase):

    def setUp(self):
        self.signal1 = SineSignal(frequency=1, amplitude=1, duration=1)
        self.signal2 = SquareSignal(frequency=1, amplitude=1, duration=1)
        self.signal3 = SawtoothSignal(frequency=1, amplitude=1, duration=1)

    def test_time_scaling(self):
        scaled_signal = time_scaling(self.signal1, 2)
        self.assertEqual(len(scaled_signal), 200)  # Assuming 100 samples for original

    def test_amplitude_scaling(self):
        scaled_signal = amplitude_scaling(self.signal1, 2)
        self.assertEqual(scaled_signal.amplitude, 2)

    def test_time_shifting(self):
        shifted_signal = time_shifting(self.signal1, 0.5)
        self.assertNotEqual(shifted_signal.time[0], self.signal1.time[0])

    def test_time_reversal(self):
        reversed_signal = time_reversal(self.signal1)
        self.assertEqual(reversed_signal.time[0], self.signal1.time[-1])

    def test_signal_addition(self):
        added_signal = signal_addition(self.signal1, self.signal2)
        self.assertEqual(len(added_signal), 100)  # Assuming both signals have the same length

    def test_signal_multiplication(self):
        multiplied_signal = signal_multiplication(self.signal1, self.signal2)
        self.assertEqual(len(multiplied_signal), 100)  # Assuming both signals have the same length

if __name__ == '__main__':
    unittest.main()