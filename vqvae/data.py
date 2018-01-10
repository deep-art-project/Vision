from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


def get_dataloader(root, train=True, download=False, shuffle=True,
				   batch_size=128, num_workers=4, pin_memory=False):
	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
		)
	dataset = CIFAR10(root, train, download=download, transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
							num_workers=num_workers, pin_memory=pin_memory)
	return dataloader