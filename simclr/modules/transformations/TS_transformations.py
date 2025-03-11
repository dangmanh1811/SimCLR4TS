import numpy as np

class Jittering:
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean

	def __call__(self, x):
		return x + np.random.normal(loc=0., scale=self.std, size=x.shape)

	def __repr__(self):
		return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


class Scaling:
	def __init__(self, sigma=0.1):
		self.sigma = sigma

	def __call__(self, x):
		n_scale = np.random.normal(loc=1, scale=self.sigma, size=(x.shape[0], x.shape[1]))
		return x * n_scale


class Flipping:
	def __init__(self, axis=1):
		self.axis = axis # on the direction of time

	def __call__(self, x):
		return np.flip(x, axis=self.axis).copy()




