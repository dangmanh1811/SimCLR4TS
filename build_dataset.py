import torch
from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
	"""
		TensorDataset with support of transform.
	"""
	def __init__(self, data, transform_A=None, transform_B=None):
		assert all(data[0].shape[0] == item.shape[0] for item in data)
		self.data = data
		self.transform_A = transform_A
		self.transform_B = transform_B

	def __getitem__(self, index):
		x = self.data[0][index]

		if self.transform_A:
			x1 = self.transform_A(x)
		else:
			x1 = x

		if self.transform_B:
			x2 = self.transform_B(x)
		else:
			x2 = x

		# x = (x1, x2)
		y = self.data[1][index]

		return torch.tensor(x1).float(), torch.tensor(x2).float(), torch.tensor(y) # np.array(x)

	def __len__(self):
		return self.data[0].shape[0]