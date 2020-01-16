import numpy as np
import pandas as pd
from mlblocks import MLBlock


class ApplyPrimitive:

    def __init__(self, primitive_name, keywords, axis, zip=None,
                 init_params=None, **hyperparameters):
        if init_params:
            init_params = init_params.copy()
            init_params.update(hyperparameters)
        else:
            init_params = hyperparameters

        self.mlblock = MLBlock(primitive_name, **init_params)
        self.keywords = keywords
        self.axis = axis
        self.zip = zip or list()

    def produce(self, X, **kwargs):
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
            X = X.values

        if self.axis == 1:
            X = X.T

        iterables = [X]
        keys = [self.keywords.get('X', 'x')]
        for key in self.zip:
            value = kwargs.pop(key)
            if isinstance(value, pd.DataFrame):
                if index is None:
                    value = value.values
                else:
                    value = value.loc[index]

            if len(value) != len(X):
                raise ValueError('Only iterables of the same length as X can be zipped to it.')

            iterables.append(value)
            keys.append(self.keywords.get(key, key))

        kwargs = {
            self.keywords.get(key, key): value
            for key, value in kwargs.items()
        }

        result = list()
        for rows in zip(*iterables):
            kwargs.update(dict(zip(keys, rows)))
            result.append(self.mlblock.produce(**kwargs))

        if index is None:
            return X
        else:
            return pd.DataFrame(result, index=index)


# class ApplyPrimitiveAxis:
# 
#     def __init__(self, primitive_name, keywords, axis, init_params=None, **hyperparameters):
#         if init_params:
#             init_params = init_params.copy()
#             init_params.update(hyperparameters)
#         else:
#             init_params = hyperparameters
# 
#         self.mlblock = MLBlock(primitive_name, **init_params)
#         self.keywords = keywords
#         self.axis = axis
# 
#     def produce(self, **kwargs):
#         iterables = list()
#         keys = list()
#         for key, axis in self.axis.items():
#             value = kwargs.pop(key)
#             if isinstance(value, pd.DataFrame):
#                 value = value.values
# 
#             if axis == 1:
#                 value = value.T
# 
#             keys.append(self.keywords.get(key, key))
#             iterables.append(value)
# 
#         kwargs = {
#             self.keywords.get(key, key): value
#             for key, value in kwargs.items()
#         }
# 
#         result = list()
#         for rows in zip(*iterables):
#             kwargs.update(dict(zip(keys, rows)))
#             result.append(self.mlblock.produce(**kwargs))
# 
#         return np.array(result)
# 
# 
# class ApplyPrimitiveX:
# 
#     def __init__(self, primitive_name, keywords, axis, init_params, **hyperparameters):
#         if init_params:
#             init_params = init_params.copy()
#             init_params.update(hyperparameters)
#         else:
#             init_params = hyperparameters
# 
#         self.mlblock = MLBlock(primitive_name, **init_params)
#         self.keywords = keywords
#         self.axis = axis
# 
#     def apply(self, x, **kwargs):
#         produce_kwargs = {
#             self.keywords['X']: x
#         }
#         for key, value in kwargs.items():
#             produce_kwargs[self.keywords.get(key, key)] =  value
# 
#         return self.mlblock.produce(**produce_kwargs)
# 
#     def produce(self, X, **kwargs):
#         dataframe = isinstance(X, pd.DataFrame)
#         if not dataframe:
#             X = pd.DataFrame(X)
# 
#         X = X.apply(self.apply, axis=self.axis, **kwargs)
# 
#         if dataframe:
#             return X
#         else:
#             return X.values
