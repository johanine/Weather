from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# transformador propio para tareas como operaciones de limpieza personalizadas o combinación de atributos específicos. 
# funcionalidades de Scikit-Learn (como las canalizaciones)
# clase de transformador que agrega los atributos:
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.rooms_ix, self.bedrooms_ix, self.population_ix, self.household_ix = 3, 4, 5, 6
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
