from arcane_dev.utils import utils
from arcane_dev.spectrum import model
import numpy as np
import matplotlib.pyplot as plt
import pandas

eps = 1.0e-6
def test_conversions(fwhm,fgfwhm):
    depth = 1.0
    gfwhm,lfwhm = utils.wxg2wgwl(fwhm, fgfwhm)
    sigma,gamma = utils.wxg2gslg(fwhm,fgfwhm)
    sigma1,gamma1 = utils.wxl2gslg(fwhm,lfwhm)
    assert np.abs(sigma-sigma1)<eps and np.abs(gamma-gamma1)<eps

    gfwhm1, lfwhm1 = utils.wxl2wgwl(fwhm,lfwhm/fwhm)
    assert np.abs(gfwhm1-gfwhm)<eps and np.abs(lfwhm1-lfwhm)<eps

    assert np.abs(gfwhm-utils.gs2wg(utils.wg2gs(gfwhm)))<eps
    assert np.abs(lfwhm-utils.lg2wl(utils.wl2lg(lfwhm)))<eps

    fwhm1,fgfwhm1,flfwhm1 = utils.wgwl2wxgxl(gfwhm,lfwhm)
    assert np.abs(fwhm1-fwhm)<eps and np.abs(fgfwhm1-fgfwhm)<eps and \
        np.abs(lfwhm - fwhm1*flfwhm1)<eps

    fwhm1,fgfwhm1,flfwhm1 = utils.gslg2wxgxl(sigma,gamma)
    assert np.abs(fwhm1-fwhm)<eps and np.abs(fgfwhm1-fgfwhm)<eps and \
        np.abs(lfwhm - fwhm1*flfwhm1)<eps


    ew = utils.voigt_EW_sigma_gamma(depth,sigma,gamma)
    ew1 = utils.voigt_EW_fwhm_flfwhm(depth,fwhm,lfwhm/fwhm)
    ew2 = utils.voigt_EW_fwhm_fgfwhm(depth,fwhm,fgfwhm)
    print(depth,fwhm,ew)
    assert np.abs(ew-ew1)<eps and np.abs(ew1-ew2)<eps

    assert np.abs(depth-utils.voigt_depth_sigma_gamma(ew,sigma,gamma))<eps and \
        np.abs(depth-utils.voigt_depth_fwhm_flfwhm(ew,fwhm,lfwhm/fwhm))<eps and \
        np.abs(depth-utils.voigt_depth_fwhm_fgfwhm(ew,fwhm,fgfwhm))<eps 

def plot_voigt_profile(x0s,depths,fwhms,fgfwhms,ax):
    gfwhm,lfwhm = utils.wxg2wgwl(fwhms, fgfwhms)
    sigma,gamma = utils.wxg2gslg(fwhms,fgfwhms)   
    fvoigt = utils.voigts_multi_fwhm_fgfwhm(x0s,depths,fwhms,fgfwhms)
    fvoigt2 = utils.voigts_multi_fwhm_flfwhm(x0s,depths,fwhms,lfwhm/fwhms)
    fvoigt3 = utils.voigts_multi_sigma_gamma(x0s,depths,sigma,gamma)

    xbin = np.linspace(np.min(x0s)-10.*(fwhms[0]),\
        np.max(x0s)+10.*(fwhms[0]),1000)
    ax.plot(xbin,fvoigt(xbin),'C0-')
    ax.plot(xbin,fvoigt2(xbin),'C1--')
    ax.plot(xbin,fvoigt3(xbin),'C2.')

def test_voigt():
    fig, axs = plt.subplots(3,1,figsize=(15,15),sharex=True,sharey=True)
    plot_voigt_profile(
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.1]),
        np.array([1.0]),
        axs[0])
    plot_voigt_profile(
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.1]),
        np.array([0.6]),
        axs[1])
    plot_voigt_profile(
        np.array([0.0,0.1]),
        np.array([1.0,0.4]),
        np.array([0.1,0.15]),
        np.array([0.6,0.8]),
        axs[2])
    fig.savefig('voigt_profiles.pdf')


def test_voigt_fit():

    fig, axs = plt.subplots(3,3,figsize=(15,15),sharex=True,sharey=True)

    fvoigt = utils.voigts_multi_fwhm_fgfwhm(\
        np.array([0.0]),np.array([0.3]),np.array([0.1]),np.array([1.0]))
    xbin = np.linspace(-1.,1.,1000)
    yy = fvoigt(xbin) + 0.01*np.random.randn(len(xbin))
    fit_mask = (-0.15<xbin)&(xbin<0.15)
    for ax in axs[0]:
        ax.plot(xbin,fvoigt(xbin),'C7-',lw=1.)
        ax.plot(xbin[fit_mask],fvoigt(xbin)[fit_mask],'C0-',ms=1.)
        ax.plot(xbin,yy,'ko',ms=1.)

    print('True',0.0,0.3,0.1,1.0)
    prof = model.LineProfile(0.0,initial_depth=0.5,initial_fwhm=0.1)
    ax = axs[0,0]
    ax.set_title('Fix FWHM_G/FWHM (force Gauss')
    prof.fit_control['voigt'] = np.array([False])
    prof.fit(xbin[fit_mask],yy[fit_mask])
    ax.plot(xbin,prof.evaluate(xbin),'C2-',lw=2.)
    ax.plot(prof.wavelength[prof.use_flag],prof.flux[prof.use_flag],'C1o',ms=2.)
    print('Fix FWHM_G/FWHM',prof.model_parameters)
    
    ax = axs[0,1]
    ax.set_title('Voigt fit')
    prof.fit_control['voigt'] = np.array([True])
    prof.fit(xbin[fit_mask],yy[fit_mask])
    ax.plot(xbin,prof.evaluate(xbin),'C2-',lw=2.)
    ax.plot(prof.wavelength[prof.use_flag],prof.flux[prof.use_flag],'C1o',ms=2.)
    print('Voigt',prof.model_parameters)

    fvoigt = utils.voigts_multi_fwhm_fgfwhm(\
        np.array([0.0,0.2]),np.array([0.5,0.3]),np.array([0.1,0.15]),np.array([0.6,0.9]))
    xbin = np.linspace(-1.,1.,1000)
    yy = fvoigt(xbin) + 0.01*np.random.randn(len(xbin))
    fit_mask = (-0.2<xbin)&(xbin<0.2+0.2)
    for ax in axs[1]:
        ax.plot(xbin,fvoigt(xbin),'C7-',lw=1.)
        ax.plot(xbin[fit_mask],fvoigt(xbin)[fit_mask],'C0-',ms=1.)
        ax.plot(xbin,yy,'ko',ms=1.)
    print('True',[0.0,0.2],[0.5,0.3],[0.1,0.15],[0.6,0.9])


    prof = model.LineProfile([0.0,0.15],initial_depth=0.5,initial_fwhm=0.1)
    ax = axs[1,0]
    ax.set_title('Fix FWHM_G/FWHM (force Gauss')
    prof.fit_control['voigt'] = np.array([False]*2)
    prof.fit(xbin[fit_mask],yy[fit_mask])
    ax.plot(xbin,prof.evaluate(xbin),'C2-',lw=2.)
    ax.plot(prof.wavelength[prof.use_flag],prof.flux[prof.use_flag],'C1o',ms=2.)
    print('Fix FWHM_G/FWHM',prof.model_parameters)
    
    ax = axs[1,1]
    ax.set_title('Voigt fit')
    prof.fit_control['voigt'] = np.array([True]*2)
    prof.fit(xbin[fit_mask],yy[fit_mask])
    ax.plot(xbin,prof.evaluate(xbin),'C2-',lw=2.)
    ax.plot(prof.wavelength[prof.use_flag],prof.flux[prof.use_flag],'C1o',ms=2.)
    print('Voigt',prof.model_parameters)



    fig.savefig('fit_voigt_profiles.pdf')


def test_voigt_fit2():

    def convert_to_pandas(profs):
        out_dict = {}
        for ii in range(profs[0].nlines):
            for key in ['center','dwvl','depth','fwhm','fgfwhm']:
                out_dict[f'{key}_{ii}'] = []
        for p in profs:            
            for key in ['center','dwvl','depth','fwhm','fgfwhm']:
                for ii,val in enumerate(p.model_parameters[key]):
                    out_dict[f'{key}_{ii}'].append(val)
        return pandas.DataFrame(out_dict) 
      

    ## Test on pure gaussians
    def test_pure_gaussian():
        fvoigt = utils.voigts_multi_fwhm_fgfwhm(\
            np.array([0.0]),np.array([0.3]),np.array([0.1]),np.array([1.0]))
        xbin = np.linspace(-1.,1.,1000)
        yy = fvoigt(xbin) + 0.01*np.random.randn(len(xbin))
        fit_mask = (-0.15<xbin)&(xbin<0.15)
        prof = model.LineProfile(0.0,initial_depth=0.5,initial_fwhm=0.1)
        prof.fit_control['voigt'] = np.array([False])
        prof.fit(xbin[fit_mask],yy[fit_mask])
        return prof
    result = convert_to_pandas([test_pure_gaussian() for ii in range(100)])
    result.to_csv('pure_gaussian_fitMC.csv')

    def test_pure_gaussian():
        fvoigt = utils.voigts_multi_fwhm_fgfwhm(\
            np.array([0.0]),np.array([0.3]),np.array([0.1]),np.array([1.0]))
        xbin = np.linspace(-1.,1.,1000)
        yy = fvoigt(xbin) + 0.01*np.random.randn(len(xbin))
        fit_mask = (-0.15<xbin)&(xbin<0.15)
        prof = model.LineProfile(0.0,initial_depth=0.5,initial_fwhm=0.1)
        prof.fit_control['voigt'] = np.array([False])
        prof.fit(xbin[fit_mask],yy[fit_mask])
        return prof
    result = convert_to_pandas([test_pure_gaussian() for ii in range(100)])
    result.to_csv('pure_gaussian_fitMC.csv')


    def test_pure_gaussian_v():
        fvoigt = utils.voigts_multi_fwhm_fgfwhm(\
            np.array([0.0]),np.array([0.3]),np.array([0.1]),np.array([1.0]))
        xbin = np.linspace(-1.,1.,1000)
        yy = fvoigt(xbin) + 0.01*np.random.randn(len(xbin))
        fit_mask = (-0.15<xbin)&(xbin<0.15)
        prof = model.LineProfile(0.0,initial_depth=0.5,initial_fwhm=0.1)
        prof.fit_control['voigt'] = np.array([True])
        prof.fit(xbin[fit_mask],yy[fit_mask])
        return prof
    result = convert_to_pandas([test_pure_gaussian_v() for ii in range(100)])
    result.to_csv('pure_gaussian_vfitMC.csv')

    def test_voigt():
        fvoigt = utils.voigts_multi_fwhm_fgfwhm(\
            np.array([0.0,0.2]),np.array([0.5,0.3]),np.array([0.1,0.15]),np.array([0.6,0.9]))
        xbin = np.linspace(-1.,1.,1000)
        yy = fvoigt(xbin) + 0.01*np.random.randn(len(xbin))
        fit_mask = (-0.2<xbin)&(xbin<0.2+0.2)
        prof = model.LineProfile([0.0,0.15],initial_depth=0.5,initial_fwhm=0.1)
        prof.fit_control['voigt'] = np.array([False]*2)
        prof.fit(xbin[fit_mask],yy[fit_mask])
        return prof
    result = convert_to_pandas([test_voigt() for ii in range(100)])
    result.to_csv('voigt_fitMC.csv')

    def test_voigt_v():
        fvoigt = utils.voigts_multi_fwhm_fgfwhm(\
            np.array([0.0,0.2]),np.array([0.5,0.3]),np.array([0.1,0.15]),np.array([0.6,0.9]))
        xbin = np.linspace(-1.,1.,1000)
        yy = fvoigt(xbin) + 0.01*np.random.randn(len(xbin))
        fit_mask = (-0.2<xbin)&(xbin<0.2+0.2)
        prof = model.LineProfile([0.0,0.15],initial_depth=0.5,initial_fwhm=0.1)
        prof.fit_control['voigt'] = np.array([True]*2)
        prof.fit(xbin[fit_mask],yy[fit_mask])
        return prof
    result = convert_to_pandas([test_voigt_v() for ii in range(1000)])
    result.to_csv('voigt_vfitMC.csv')


if __name__ == '__main__':
    test_conversions(1.0,0.8)
    test_conversions(1.0,0.0)
    test_conversions(1.0,1.0)
    test_voigt()
    test_voigt_fit()
    test_voigt_fit2()