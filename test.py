import dataset_HAR
from build_dataset import CustomTensorDataset
from simclr.modules.transformations import Jittering, Scaling, Flipping


train_dataset = CustomTensorDataset(
	data=(dataset_HAR.train_x, dataset_HAR.train_y),
	transform_A=Jittering(0, 0.1)
	)


x1, x2, y = train_dataset[0]
print(x1.shape, x2.shape, y.shape)
print(x2)