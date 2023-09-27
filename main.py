import tkinter as tk
from youAreArt_colorSmall import run_yarcs
from youAreArt_colorLarge import run_yarcl
from youAreArt_colorFrequent_small import run_yarcfs
from youAreArt_colorFrequent_large import run_yarcfl

from lineArtLiveChangingColor import run_lalcc
from lineArtLiveChanging_ColorFrequent import run_lalccf
from lineArtLive_blackWhite import run_lalbw

from dynamic import actual_run

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

        #Add a separator
        self.label5 = tk.Label(master, text="---------------------------", font=('Arial', 10))
        self.label5.pack(pady=(10,10))

        self.label6 = tk.Label(master, text="Face detection w/ color change per detection", font=('Arial', 10))
        self.label6.pack()

        self.button5 = tk.Button(master, text="Run LALCC", command=run_lalcc)
        self.button5.pack(pady=(0,10))

        self.label7 = tk.Label(master, text="Face detection w/ color change per frame", font=('Arial', 10))
        self.label7.pack()

        self.button6 = tk.Button(master, text="Run LALCCF", command=run_lalccf)
        self.button6.pack(pady=(0,10))

        self.label8 = tk.Label(master, text="Face detection w/ black and white", font=('Arial', 10))
        self.label8.pack()

        self.button7 = tk.Button(master, text="Run LALBW", command=run_lalbw)
        self.button7.pack(pady=(0,10))

        #Add a separator
        self.label5 = tk.Label(master, text="---------------------------", font=('Arial', 10))
        self.label5.pack(pady=(10,10))

        self.label9 = tk.Label(master, text="Dynamic", font=('Arial', 10))
        self.label9.pack()

        self.button8 = tk.Button(master, text="Run Dynamic", command=actual_run)
        self.button8.pack(pady=(0,10))

        

def main():
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
