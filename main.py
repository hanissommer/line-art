import tkinter as tk
from youAreArt_colorSmall import run_yarcs
from youAreArt_colorLarge import run_yarcl
from youAreArt_colorFrequent_small import run_yarcfs
from youAreArt_colorFrequent_large import run_yarcfl

class SimpleApp:
    def __init__(self, master):
        self.master = master
        master.title('You are the Art')
        
        self.label1 = tk.Label(master, text="Body detection w/ color change per detection (s)", font=('Arial', 10))
        self.label1.pack()
        
        self.button1 = tk.Button(master, text="Run YARCS", command=run_yarcs)
        self.button1.pack(pady=(0,10))
        
        self.label2 = tk.Label(master, text="Body detection w/ color change per detection (l)", font=('Arial', 10))
        self.label2.pack()
        
        self.button2 = tk.Button(master, text="Run YARCL", command=run_yarcl)
        self.button2.pack(pady=(0,10))
        
        self.label3 = tk.Label(master, text="Body detection w/ color change per frame (s)", font=('Arial', 10))
        self.label3.pack()
        
        self.button3 = tk.Button(master, text="Run YARCFS", command=run_yarcfs)
        self.button3.pack(pady=(0,10))
        
        self.label4 = tk.Label(master, text="Body detection w/ color change per frame (l)", font=('Arial', 10))
        self.label4.pack()
        
        self.button4 = tk.Button(master, text="Run YARCFL", command=run_yarcfl)
        self.button4.pack(pady=(0,10))

def main():
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
