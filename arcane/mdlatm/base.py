class ModelAtm(dict):
    '''
    Base class for model atmospheres.  This class is a dictionary of
    atmospheric properties.
    '''
    def __init__(self, *args, **kwargs):
        super(ModelAtm, self).__init__(*args, **kwargs)