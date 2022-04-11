import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms

from laplace.utils import ModuleNameSubnetMask
from laplace import Laplace
from laplace.curvature import AsdlGGN, AsdlEF, BackPackGGN, BackPackEF

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.l1 = torch.nn.Linear(784, 10)
		self.l2 = torch.nn.Linear(10, 100)
		self.l3 = torch.nn.Linear(100, 2)
		self.l4 = torch.nn.Linear(2, 10)

		self.tanh = torch.nn.Tanh()

	def forward(self, x):
		x = x.reshape(x.shape[0], -1)
		x = self.tanh(self.l1(x))
		x = self.tanh(self.l2(x))
		x = self.tanh(self.l3(x))
		x = self.tanh(self.l4(x))
		return x


model1 = torch.nn.Sequential(
			torch.nn.Flatten(2),
            torch.nn.Linear(784, 10), 
            torch.nn.Tanh(), 
            torch.nn.Linear(10, 100), 
            torch.nn.Tanh(), 
            torch.nn.Linear(100, 50), 
            torch.nn.Tanh(), 
            torch.nn.Linear(50, 10),
        ).cuda()
model2 = Model().cuda()

dataset = datasets.MNIST(root='~/Code/bayesian-lottery-tickets/data', train=True, download=False, transform=transforms.ToTensor())
dataset = torch.utils.data.Subset(dataset, range(1000))
#dataset = [(d[0].reshape(1, -1), d[1]) for d in dataset]
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
                
#for backend in [AsdlGGN, AsdlEF, BackPackGGN, BackPackEF]:
for backend in [BackPackGGN, BackPackEF]:
	for i, model in enumerate([model1, model2]):
		if i == 1:
			continue
		#for hess in ['full', 'diag']:
		for hess in ['diag']:
			print(f'{backend.__name__} - model{i+1} - {hess}')
			if i == 0:
				subnetwork_mask = ModuleNameSubnetMask(model1, module_names=['7'])
			else:
				subnetwork_mask = ModuleNameSubnetMask(model2, module_names=['l4'])
			subnetwork_indices = subnetwork_mask.select()

			la = Laplace(model, 'classification',
						subset_of_weights='subnetwork',
						hessian_structure=hess,
						subnetwork_indices=subnetwork_indices,
						backend=backend)
			la.fit(loader)
			samples = la.sample(3)
			la.optimize_prior_precision(method='marglik')