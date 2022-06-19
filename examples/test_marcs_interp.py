from arcane.mdlatm import marcs
import matplotlib.pyplot as plt
import numpy as np

model_sun_all = marcs.get_marcs_mod(5777.,4.40,0.0,alphafe=0.01,check_interp=True)
model_sun_none_all = marcs.get_marcs_mod(5777.,4.40,0.0,alphafe=None,check_interp=True)

model_sun = model_sun_all['000']
model_sun_none = model_sun_none_all['000']

sun_marcs = marcs.read_marcs(marcs.data_dir+'sun.mod')

inlog = ['Pe','Pg','Prad','Pturb','KappaRoss','Density','Vconv','RHOX']

clms = []
for key ,val in sun_marcs.items():
  if type(val) is np.ndarray:
    clms.append(key)

nclm = (len(clms)-1)//3+1
fig, axs = plt.subplots(3,nclm,figsize=(5*nclm,15))
axs = axs.ravel()
for ii,clm in enumerate(clms):
  plt.sca(axs[ii])
  if clm in inlog:
    plt.plot(sun_marcs['lgTauR'],np.log10(sun_marcs[clm]),'k-')
    plt.plot(model_sun['lgTauR'],np.log10(model_sun[clm]),'C0-')
    plt.plot(model_sun_none['lgTauR'],np.log10(model_sun_none[clm]),'C0--')
  else:
    plt.plot(sun_marcs['lgTauR'],sun_marcs[clm],'k-')
    plt.plot(model_sun['lgTauR'],model_sun[clm],'C0-')
    plt.plot(model_sun_none['lgTauR'],model_sun_none[clm],'C0--')

  plt.xlabel('lgTauR')
  plt.ylabel(clm)
fig.tight_layout()
fig.savefig('MARCS_interp_test0.pdf')


model_sun_all = marcs.get_marcs_mod(5777.,4.40,0.05,alphafe=0.01,check_interp=True)

fig, axs = plt.subplots(3,nclm,figsize=(5*nclm,15))
axs = axs.ravel()
for ii,clm in enumerate(clms):
  plt.sca(axs[ii])
  for mdlkey in ['111','112','121','122','211','212','221','222']:
    model1 = model_sun_all[mdlkey]
    if clm in inlog:
      plt.plot(model1['lgTauR'],np.log10(model1[clm]),'C7-',lw=0.5)
    else:
      plt.plot(model1['lgTauR'],model1[clm],'C7-',lw=0.5)
  for mdlkey in ['110','120','210','220']:
    model1 = model_sun_all[mdlkey]
    if clm in inlog:
      plt.plot(model1['lgTauR'],np.log10(model1[clm]),'k-',lw=0.5)
    else:
      plt.plot(model1['lgTauR'],model1[clm],'k-',lw=0.5)
  for mdlkey in ['100','200']:
    model1 = model_sun_all[mdlkey]
    if clm in inlog:
      plt.plot(model1['lgTauR'],np.log10(model1[clm]),'C0-',lw=1.0)
    else:
      plt.plot(model1['lgTauR'],model1[clm],'C0-',lw=1.0)

  model1 = model_sun_all['000']
  if clm in inlog:
    plt.plot(model1['lgTauR'],np.log10(model1[clm]),'C1-',lw=2.0)
  else:
    plt.plot(model1['lgTauR'],model1[clm],'C1-',lw=2.0)

  plt.xlabel('lgTauR')
  plt.ylabel(clm)
fig.tight_layout()
fig.savefig('MARCS_interp_test1p.pdf')


model_sun_all = marcs.get_marcs_mod(5225.,3.30,0.00,alphafe=0.01,check_interp=True)

fig, axs = plt.subplots(3,nclm,figsize=(5*nclm,15))
axs = axs.ravel()
for ii,clm in enumerate(clms):
  plt.sca(axs[ii])
  for mdlkey in ['111','112','121','122','211','212','221','222']:
    model1 = model_sun_all[mdlkey]
    if clm in inlog:
      plt.plot(model1['lgTauR'],np.log10(model1[clm]),'C7-',lw=0.5)
    else:
      plt.plot(model1['lgTauR'],model1[clm],'C7-',lw=0.5)
  for mdlkey in ['110','120','210','220']:
    model1 = model_sun_all[mdlkey]
    if clm in inlog:
      plt.plot(model1['lgTauR'],np.log10(model1[clm]),'k-',lw=0.5)
    else:
      plt.plot(model1['lgTauR'],model1[clm],'k-',lw=0.5)
  for mdlkey in ['100','200']:
    model1 = model_sun_all[mdlkey]
    if clm in inlog:
      plt.plot(model1['lgTauR'],np.log10(model1[clm]),'C0-',lw=1.0)
    else:
      plt.plot(model1['lgTauR'],model1[clm],'C0-',lw=1.0)

  model1 = model_sun_all['000']
  if clm in inlog:
    plt.plot(model1['lgTauR'],np.log10(model1[clm]),'C1-',lw=2.0)
  else:
    plt.plot(model1['lgTauR'],model1[clm],'C1-',lw=2.0)

  plt.xlabel('lgTauR')
  plt.ylabel(clm)
fig.tight_layout()
fig.savefig('MARCS_interp_test1s.pdf')
