from arcane.mdlatm import marcs
import numpy as np

np.random.seed(0)
flst = np.random.choice(marcs.grid['filename'].values,100)


for f in flst:
  print(f)
  model1 = marcs.read_marcs(marcs.data_dir+f)
  marcs.write_marcs('temp.mod',model1)
  model2 = marcs.read_marcs('temp.mod')
  for key in model1.keys():
    if not key in model2.keys():
      print(f'{key} is not in temp.mod!!')
    if type(model1[key]) is str:
      continue
    elif type(model1[key]) is dict:
      for key2 in model1[key].keys():
        if not key2 in model2[key].keys():
          print(f'{key2} is not present in {key} of temp.mod')
        if np.abs(model1[key][key2]-model2[key][key2])>0.0:
          print(f'the models are different in {key2} in {key}, by '+\
            '{0:.3f}'.format(np.abs(model1[key][key2]-model2[key][key2])))
      for key2 in model2[key].keys():
        if not key2 in model1[key].keys():
          print(f'{key2} is not present in {key} of the original model')
    elif np.max(np.abs(model1[key]-model2[key]))>0.0:
          print(f'the models are different in {key} by at most '+\
            '{0:.3f}'.format(np.abs(model1[key]-model2[key])))
  for key in model2.keys():
    if not key in model1.keys():
      print(f'{key} is not in the original model!!')


