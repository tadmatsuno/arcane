from arcane.utils import solarabundance as sa
from arcane.utils import isotopic_ratio as ir
import matplotlib.pyplot as plt
import numpy as np

sa.load_solarabundance("Lodders2009")
L09 = sa.solar_abundance.copy()
sa.load_solarabundance("Asplund2009")
A09 = sa.solar_abundance.copy()
sa.load_solarabundance("Asplund2021")
A21 = sa.solar_abundance.copy()
sa.load_solarabundance("Bergemann2025")
B25 = sa.solar_abundance.copy()

def plot_abundance_one(ab):
    return np.nonzero(ab>-90)[0]+1,ab[ab>-90]

plt.plot(*plot_abundance_one(L09),"o-",label="Lodders2009")
plt.plot(*plot_abundance_one(A09),"s-",label="Asplund2009")
plt.plot(*plot_abundance_one(A21),"^-",label="Asplund2021")
plt.plot(*plot_abundance_one(B25),"d-",label="Bergemann2025")
#plt.xlim(5,10)
plt.legend()

ir.load_isotopic_ratio("solar_Lodders2009")
L09i = ir.isotopic_ratio.copy()
ir.load_isotopic_ratio("solar_Asplund2021")
A21i = ir.isotopic_ratio.copy()
ir.load_isotopic_ratio("solar_Bergemann2025")
B25i = ir.isotopic_ratio.copy()
def plot_isotopic_ratio_one(iratio, symb):
    for z in range(50,60):
        amasses = np.nonzero(iratio[z,:]>1)[0]
        fracs = iratio[z,amasses]
        plt.plot(amasses,fracs,f"C{z-1}{symb}-")
plot_isotopic_ratio_one(L09i,"o")
plot_isotopic_ratio_one(A21i,"s")
plot_isotopic_ratio_one(B25i,"^")