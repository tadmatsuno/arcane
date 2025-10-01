from scipy.interpolate import CubicSpline
import numpy as np

class ModelAtm(dict):
    '''
    Base class for model atmospheres.  This class is a dictionary of
    atmospheric properties.
    '''
    def __init__(self, *args, **kwargs):
        super(ModelAtm, self).__init__(*args, **kwargs)
    def copy(self):
        return type(self)(self)
    def resample_model(self,lgtauRnew, inplace=False):
        new_model = self.copy()
        new_model['ndepth'] = len(lgtauRnew)
        new_model['lgTauR'] = lgtauRnew
        for key in self.keys():
            if (type(self[key]) is np.ndarray) and (key != 'lgTauR'):
                cs = CubicSpline(self['lgTauR'],self[key])
                new_model[key] = cs(new_model['lgTauR'])
        if inplace:
            self.clear()
            self.update(new_model)
            return
        else:
            return new_model