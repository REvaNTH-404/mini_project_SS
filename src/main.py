import tkinter as tk
from gui import SignalGUI

def main():
    root = tk.Tk()
    app = SignalGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
