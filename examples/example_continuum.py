from astropy.table import Table
import sys
from arcane.continuum import continuum
import requests
import os 
if not os.path.exists('readmultispec.py'):
  print('Need readmultispec.py by kgullikson88')
  r = requests.get('https://github.com/kgullikson88/General/raw/master/readmultispec.py',
    allow_redirects=True)
  open('readmultispec.py','wb').write(r.content)
  print('You probably need to make sure that the script works in python 3.X')
  exit
import readmultispec

if __name__ == '__main__':
  print("This should launch a new graphical window")
  print("Once it is launced you can change parameters interactively")
  print("For a long1d format, such as UVES spectra from ESO archive,")
  print("a spectrum is split into small pieces so that you can slightly")
  print("modify your approach depending on S/N and line densities at ")
  print("different wavelength.")
  print("You can also use multispec format of iraf\n")

  print("In the graphical window, a few shortcut keys are prepared")
  print("[n] to move to a different region")
  print("[f] to refit")
  print("[s]->[s] to specify the regions to be used for fitting")
  print("         the x-coordinate of the cursor location is used as boundary")
  print("[t] to clear all the wavelength selection")
  print("You can change other parameters from the boxes")

  try:
    assert sys.argv[1] in ['long1d','multi']
    if sys.argv[1] == 'long1d':
      test_multi_spec = False
    elif sys.argv[1] == 'multi' :
      test_multi_spec = True
  except:
    raise ValueError('python test.py [long1d/multi]\n')   
  if not os.path.exists('result'):
      os.mkdir('result')
  if test_multi_spec:
    HD122563R = readmultispec.readmultispec('data/HD122563R.fits')
    if not os.path.exists('result/HD122563_multi'):
      os.mkdir('result/HD122563_multi')
    continuum.start_gui(
      HD122563R['wavelen'][0:5],HD122563R['flux'][0:5],\
      outfile = 'result/HD122563Rn_testmulti.csv',
      form = 'multi',
      output_multi_head = 'result/HD122563_multi/')
  else:
    HD122563UVES = Table.read('data/HD122563_UVES.fits')
    test_region = (5000<HD122563UVES['WAVE'][0])&(HD122563UVES['WAVE'][0]<6000)
    continuum.start_gui(
      HD122563UVES['WAVE'][0][test_region],HD122563UVES['FLUX'][0][test_region],\
      outfile = 'result/HD122563Rn_testlong1d.csv',
      form = 'long1d')
