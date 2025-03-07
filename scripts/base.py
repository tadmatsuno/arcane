from arcane_dev.spectrum import model
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtWidgets, QtCore

class ShowSpectrumFit(QtWidgets.QMainWindow):
    def __init__(self):
        self.setWindowTitle('Spectrum')
    
    def plot_spectrum(self,model):
        '''
        Model can be a arcane_dev.model.ModelBase class or dictionary containing 'line' and 'continuum'
        '''

        
def show_used_pixels():
    pass