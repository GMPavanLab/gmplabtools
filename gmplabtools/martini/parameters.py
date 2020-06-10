import functools
from types import SimpleNamespace

import pkgutil
import io
import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader

from .data.non_bonded import ORIGINAL


def cache():
	def wrapper(method):
		@functools.wraps(method)
		def wrapped(self, *args, **kwargs):
		    return method(self, *args, **kwargs)
		return functools.lru_cache(maxsize=1)(wrapped)
	return wrapper


class Martini(SimpleNamespace):

    def __init__(self, **kwargs):
        super().__init__(**{**ORIGINAL, **kwargs})

    @property
    def env(self):
        package_loader = PackageLoader(__name__, 'sim_data')
        templateEnv = Environment(
            loader=package_loader,
        )
        return templateEnv

    def get_template(self, template, **kwargs):
        input_data = {**self.__dict__, **kwargs}
        template = self.env.get_template(template)
        return template.render(input_data)

class Param:

	non_bonded_strength = {
		'supra attractive': 0,
		'attractive': 1,
		'almost attractive': 2,
		'semi attractive': 3,
		'intermediate': 4,
		'almost intermediate': 5,
		'semi repulsive': 6,
		'almost repulsive': 7,
		'repulsive': 8,
		'super repulsive': 9
	}

	def __init__(self):
		self.f = self.interp()

	@classmethod
	def get_data(cls, filename):
		param = pkgutil.get_data('sim_data', filename)
		return pd.read_csv(io.BytesIO(param))

	@property
	@cache()
	def df(self):
		df = Param.get_data('martini-non-bonded.csv')
		df['type'] = df['type'].map(Param.non_bonded_strength)
		return df

	def min(self, field, subset=None):
		mask = self.df[field] > 0
		if subset is not None:
		    mask &= self.df['type'] == subset
		return self.df.loc[mask, field].min()

	def max(self, field, subset=None):
		mask = self.df[field] > 0
		if subset is not None:
			mask &= self.df['type'] == subset
		return self.df.loc[mask, field].max()

	def interp(self):
		df = self.df
		n = df.shape[0]
		x, y = df['c6'].values, df['c12'].values
		m =  (np.sum(x * y) * n - x.sum() * y.sum()) / (n * (x ** 2).sum() - x.sum() ** 2)
		b = y.mean() - m * x.mean()
		return lambda x: x * m + b

	def __call__(self, subset=None):
		m, M = self.min(field='c6', subset=subset), self.max(field='c6', subset=subset)
		c6 = np.random.uniform(m, M)
		return c6, self.f(c6)


def init_params(fields):
	params = Param()
	new_params = {}
	for ljparam in fields:
		vals = params()
		new_params[ljparam + '_c6'] = vals[0]
		new_params[ljparam + '_c12'] = vals[1]

	return {**ORIGINAL, **new_params}
