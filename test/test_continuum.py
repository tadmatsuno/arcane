from arcane.spectrum import model
import iofiles
import numpy as np
import matplotlib.pyplot as plt

def test_continuum():
    continuum = model.ContinuumSpline3(20.,low_rej=1.8)
    arcturus = iofiles.readspip('./DATA/arcturus.text')
    mask = (5000<arcturus['wvl'])&(arcturus['wvl']<5100)
    wvl = arcturus['wvl'][mask].values
    flx = arcturus['flx'][mask].values
    mock_flx = flx*(-(wvl-5040)**2 + 7200)
    continuum.fit(wvl,mock_flx)
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(continuum.wavelength,continuum.flux,'C7o',ms=1.)
    ax.plot(continuum.wavelength[continuum.use_flag],
        continuum.flux[continuum.use_flag],'C1o',ms=2.)
    ax.plot(continuum.wavelength,
        continuum.yfit,'C2-',ms=2.)
    ax.plot(wvl,mock_flx/flx,'k--')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('flux')
    fig.tight_layout()
    fig.savefig('./output/continuum_test.pdf')

    continuum2 = model.ContinuumSpline3(20.,low_rej=1.8,naverage=2)
    continuum2.fit(wvl,mock_flx)

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(continuum2.wavelength,continuum2.flux,'C7o',ms=1.)
    ax.plot(continuum2.wavelength[continuum2.use_flag],
        continuum2.flux[continuum2.use_flag],'C1o',ms=2.)
    ax.plot(continuum2.wavelength,
        continuum2.yfit,'C2-',ms=2.)
    ax.plot(wvl,mock_flx/flx,'k--')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('flux')
    fig.tight_layout()
    fig.savefig('./output/continuum_test_binning.pdf')

    continuum3 = model.ContinuumPolynomial(1,low_rej=3,naverage=2)
    continuum3.samples = [[6642.7,6643.3],[6643.9,6644.2]]
    mask = (6640<arcturus['wvl'])&(arcturus['wvl']<6650)
    wvl = arcturus['wvl'][mask].values
    flx = arcturus['flx'][mask].values
    mock_flx = flx*(-3.*(wvl-6645) + 100)
    continuum3.fit(wvl,mock_flx)

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(continuum3.wavelength,continuum3.flux,'C7o',ms=1.)
    ax.plot(continuum3.wavelength[continuum3.use_flag],
        continuum3.flux[continuum3.use_flag],'C1o',ms=2.)
    ax.plot(continuum3.wavelength,
        continuum3.yfit,'C2-',ms=2.)
    ax.plot(wvl,mock_flx/flx,'k--')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('flux')
    fig.tight_layout()
    fig.savefig('./output/continuum_test_poly_line.pdf')

if __name__ == '__main__':
    test_continuum()
