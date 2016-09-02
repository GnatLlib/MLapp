import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
"""
Simple demonstration and visualisation of machine learning using sci-kit learn
with gui built with tkinter

"""



import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


#Defining classifiers
ppn = Perceptron(n_iter=40, eta0=0.1, random_state =0)

lr = LogisticRegression(C= 1000.0, random_state=0)

svm = SVC(kernel='linear', C=1.0, random_state=0)


#Global figure to be used for tkinter canvas
f = Figure(figsize=(5,4), dpi=140)
a = f.add_subplot(111)


#Function to plot decision regions determined by sci-kit learn
def plot_decision_regions(X,y,classifier,test_idx, resolution=0.02):
    
    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen','gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    
    a.contourf(xx1,xx2, Z, alpha=0.4, cmap=cmap)
        
    

    #plot all samples
    for idx, cl in enumerate(np.unique(y)):
       a.scatter(x=X[y==cl,0], y =X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)
       
    #highlight test samples
    """
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        a.scatter(X_test[:,0],X_test[:,1], c='', alpha = 1.0, linewidths=1, marker = 'o', s=55, label='test set')
    """


LARGE_FONT = ("Arial", 12)

#track if file is being read
wasFileRead = False
    
#Base app for tkinter gui
class MLapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "MLapp")

        global d1, d2, d3, var1, var2, var3
        d1 = tk.StringVar(self)
        d1.set("empty")
        d2 = tk.StringVar(self)
        d2.set("empty")
        d3 = tk.StringVar(self)
        d3.set("empty")
        var1 = tk.IntVar(self)
        var1.set(0)
        var2 = tk.IntVar(self)
        var2.set(0)
        var3 = tk.IntVar(self)
        var3.set(0)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight = 1)

        self.frames = {}

        for F in (StartPage, PageOne, GraphDisplay):
            frame = F(container, self)

            self.frames[F] = frame
            

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

#Starting Frame
class StartPage(tk.Frame):

    
    def get_fields(self):
        d1.set(self.e1.get())
        d2.set(self.e2.get())
        d3.set(self.e3.get())
        wasFileRead= false

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Enter Your Data", font=LARGE_FONT)
        label.grid(row=0)

        tk.Label(self, text="Attribute 1 Data").grid(row=1)
        tk.Label(self, text="Attribute 2 Data").grid(row=2)
        tk.Label(self, text="Classification Data").grid(row=3)
        self.e1 = tk.Entry(self)
        self.e1.grid(row=1, column=1)
        self.e2 = tk.Entry(self)
        self.e2.grid(row=2, column=1)
        self.e3 = tk.Entry(self)
        self.e3.grid(row=3, column=1)

        button1 = ttk.Button(self, text = "Open File",
                            command = lambda: controller.show_frame(PageOne))

        button2 = ttk.Button(self, text = "Results Page",
                            command = lambda: controller.show_frame(GraphDisplay))

        button3 = ttk.Button(self, text ="Enter",
                             command=self.get_fields)

      
        button1.grid(row=4)
        button2.grid(row=5)
        button3.grid(row=6)

        global var1, var2, var3
       
        check1 = ttk.Checkbutton(self,text="PPN",variable=var1, onvalue=1, offvalue=0)
        check1.grid(row=7, column=0)
        check2 = ttk.Checkbutton(self,text="LR",variable=var2, onvalue=1, offvalue=0)
        check2.grid(row=7, column=1)
        check3 = ttk.Checkbutton(self,text="SVM",variable=var3, onvalue=1, offvalue=0)
        check3.grid(row=7, column=2)

#Page for opening file
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Open File", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        
        button1 = ttk.Button(self, text = "Back",
                            command = lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text="Choose File", command=self.askopenfile)
        button2.pack()

        self.file_opt = options = {}
        options['defaultextension'] = '.txt'
        options['filetypes'] = [('excel file', '.xlsx'), ('text file', '.txt')]
        options['initialdir'] = 'C:\\'
        options['parent'] = parent
        options['title'] = 'Data'
        
    def askopenfile(self):

        global file
        file = filedialog.askopenfile(mode = 'r', **self.file_opt)
        global content1, content2, content3, wasFileRead

        content1 = file.readline()
        content2 = file.readline()
        content3 = file.readline()
        wasFileRead = True
        
     
        
#Page to manage data processing and result display
class GraphDisplay(tk.Frame):

    
    def generateGraph(self,c, b):
        f.clf()
        global a
        a = f.add_subplot(111)
        

        X_train, X_test, y_train, y_test = train_test_split(c,b,test_size = 0.3, random_state = 0)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        if var1.get() == 1:
            ppn.fit(X_train_std, y_train)
            plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))
        if var2.get() == 1:
            lr.fit(X_train_std, y_train)
            plot_decision_regions(X=X_combined_std,y=y_combined,classifier=lr,test_idx=range(105,150))
        if var3.get() == 1:
            svm.fit(X_train_std, y_train)
            plot_decision_regions(X=X_combined_std,y=y_combined,classifier=svm,test_idx=range(105,150))
        
        self.canvas.draw()
    def parseData(self):
        global wasFileRead

        if wasFileRead:
            list1 = content1.split(' ')
            list2 = content2.split(' ')
            list3 = content3.split(' ')
            wasFileRead = False
        else:
            list1 = d1.get().split(' ')
            list2 = d2.get().split(' ')
            list3 = d3.get().split(' ')

        
        
        list1 = [float(i) for i in list1]
        list2 = [float(i) for i in list2]
        list3 = [int(i) for i in list3]
   
        
        features = np.column_stack((list1, list2))
    
        
        self.generateGraph(features, list3)

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Results", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text = "Back",
                            command = lambda: controller.show_frame(StartPage))
        button1.pack()

        button4 = ttk.Button(self, text="Analyze",
                             command = self.parseData)
        button4.pack()

        self.canvas = FigureCanvasTkAgg(f, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill = tk.BOTH, expand = True)

        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

      

app = MLapp()

app.mainloop()
