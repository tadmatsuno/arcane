import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QLineEdit, 
                             QPushButton, QWidget, QLabel, QHBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from arcane_dev.synthesis import readvald

class DataFramePlotter(QWidget):
    def __init__(self, df):
        super().__init__()
        
        self.df = df
        self.filtered_df = df
        
        self.initUI()
        
    def initUI(self):
        self.layout = QVBoxLayout()
        
        # Text box for input
        self.textbox = QLineEdit(self)
        self.textbox.setPlaceholderText("Species")
        self.layout.addWidget(self.textbox)
        
        # Button to apply filter
        self.filter_button = QPushButton("Filter", self)
        self.filter_button.clicked.connect(self.apply_filter)
        self.layout.addWidget(self.filter_button)
        
        # Plot area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Second plot area
        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.layout.addWidget(self.canvas2)
        
        self.setLayout(self.layout)
        
        self.plot_data()
    
    def apply_filter(self):
        value = self.textbox.text()
        self.filtered_df = self.df[self.df['species'] == value]
        self.plot_data()
    
    def plot_data(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.scatter(self.filtered_df['loggf']-self.filtered_df['expot'], self.filtered_df['depth'])
        ax.set_xlabel('loggf - expot')
        ax.set_ylabel('depth')
        
        self.canvas.draw()
        
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click)
    
    def on_click(self, event):
        if event.inaxes is not None:
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            
            # Find the closest point
            distances = np.sqrt((self.filtered_df['B'] - x) ** 2 + (self.filtered_df['C'] - y) ** 2)
            min_index = distances.idxmin()
            
            d_value = self.filtered_df.loc[min_index, 'D']
            
            self.update_secondary_plot(d_value)
    
    def update_secondary_plot(self, d_value):
        self.figure2.clear()
        ax2 = self.figure2.add_subplot(111)
        
        # Call function f(D)
        y_data = self.f(d_value)
        
        ax2.plot(y_data)
        ax2.set_title(f"Plot for D={d_value}")
        
        self.canvas2.draw()
    
    def f(self, d):
        # Example function: y = d * sin(x)
        x = np.linspace(0, 10, 100)
        return d * np.sin(x)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Example DataFrame
    df = readvald.read_vald(sys.argv[1])
    
    main = DataFramePlotter(df)
    main.show()
    
    sys.exit(app.exec_())
