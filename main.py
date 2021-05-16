import pdb
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchdiffeq as deq
from scipy.integrate import solve_ivp

'''
Base class
'''
class NODESpectralExperiment:
	def __init__(self, 
			ndim: int,
			dydt: Callable,
			y0_bounds: np.ndarray,
			batch_size: int,
			t_span: np.ndarray,
			model: nn.Module,
			lr: float=0.001,
			momentum: float=0.9,
			n_epochs: int=1000,
			loss: Callable=None,
			n_snaps: int=5,
		):
		assert len(t_span) >= 2
		assert ndim > 0
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.ndim = ndim
		self.dydt = dydt
		self.y0_bounds = y0_bounds
		self.batch_size = batch_size
		self.t_span = t_span
		self.t_span_torch = torch.from_numpy(t_span).to(self.device)
		self.model = model.double().to(self.device)
		# self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
		self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		self.n_epochs = n_epochs
		self.loss = nn.MSELoss() if loss is None else loss
		self.n_snaps = n_snaps

	def ground_truth(self, y0):
		sol = solve_ivp(self.dydt, (self.t_span[0], self.t_span[-1]), y0, t_eval=self.t_span, method='LSODA')
		return sol.y.T

	def hypothesis(self, y0):
		return deq.odeint(self.model, y0, self.t_span_torch).to(self.device)

	def get_batch(self):
		batch_y0, batch_y = [], []
		for _ in range(self.batch_size):
			y0 = np.random.uniform(low=self.y0_bounds[:,0], high=self.y0_bounds[:,1], size=self.ndim)
			y = self.ground_truth(y0)
			batch_y0.append(y0)
			batch_y.append(y)
		batch_y0 = torch.from_numpy(np.array(batch_y0)).to(self.device)
		batch_y = torch.from_numpy(np.array(batch_y)).to(self.device)
		return batch_y0, batch_y

	def train_epoch(self, y0=None, y=None):
		self.optimizer.zero_grad()
		if y0 is None:
			y0, y = self.get_batch()
		pred_y = self.hypothesis(y0).transpose(0, 1)
		loss = self.loss(pred_y, y)
		loss_val = loss.item()
		loss.backward()
		self.optimizer.step()
		return loss_val

	def plot_ground_truth(self, y0):
		assert self.ndim == 2
		y = self.ground_truth(y0)
		plt.plot(y[:,0], y[:,1])
		plt.show()

	def run_and_visualize(self):
		loss_history = []
		snaps = []
		y0, y = self.get_batch()
		# pdb.set_trace()
		for i in range(self.n_epochs):
			loss = self.train_epoch(y0=y0, y=y)
			print(f'Epoch: {i} Loss: {loss}')
			loss_history.append(loss)
			with torch.no_grad():
				if i == 0 or i % int((self.n_epochs+1) / (self.n_snaps-1)) == 0:
					pred_y = self.hypothesis(y0[0])
					snaps.append(pred_y)

		fig, axs = plt.subplots(nrows=1, ncols=1+self.n_snaps, figsize=(20, 4))
		axs[0].plot(loss_history)
		axs[0].set_xlabel('Epoch')
		axs[0].set_ylabel('MSE Loss')
		for i, pred_y in enumerate(snaps):
			axs[i+1].plot(y[0,:,0], y[0,:,1], color='green', label='ground truth')
			axs[i+1].plot(pred_y[:,0], pred_y[:,1], color='red', label='hypothesis')

		plt.tight_layout()
		plt.show()


'''
Van der Pol oscillator
'''
def vdp_experiment():
	np.random.seed(1000)
	ndim = 2
	mu = 3.
	y0_bounds = np.array([[-2, 2], [-4, 4]])
	batch_size = 1000
	t_span = np.linspace(0, 15, 500)

	def dydt(t, y):
		return np.array([y[1], mu*(1-y[0]**2)*y[1] - y[0]])

	class Model(nn.Module):
		def __init__(self):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(2, 20),
				nn.ELU(),
				nn.Linear(20, 30),
				nn.Tanh(),
				nn.Linear(30, 30),
				nn.Tanh(),
				nn.Linear(30, 20),
				nn.ELU(),
				nn.Linear(20, 2),
			)
			for m in self.net.modules():
				if isinstance(m, nn.Linear):
					nn.init.normal_(m.weight, mean=0, std=0.1)
					nn.init.constant_(m.bias, val=0)

		def forward(self, t, y):
			return self.net(y)

	model = Model()
	experiment = NODESpectralExperiment(ndim, dydt, y0_bounds, batch_size, t_span, model, n_epochs=500, lr=0.001)
	# experiment.plot_ground_truth()
	experiment.run_and_visualize()

'''
Duffing equation
'''
# def system(alpha, beta, gamma, delta, u):
# 	def f(t, y):
# 		dydt = np.zeros_like(y)
# 		dydt[0] = y[1]
# 		dydt[1] = -delta*y[1] - alpha*y[0] - beta*(y[0]**3) + gamma*u(t)
# 		return dydt
# 	return f

# def dataset(t0: float=0., tf: float=400, n=8000, alpha=-1.0, beta=1.0, gamma=0.5, delta=0.3, x0=-1.0, y0=2.0, u=lambda t:0.):
# 	"""Duffing oscillator 
	
# 	Args:
# 		tmax: # seconds 
# 		n: # data points (dt = tmax / n)
# 		x0: initial condition
# 		y0: initial condition
# 		u: control signal (Callable : time -> float)
# 	"""
# 	t = np.linspace(t0, tf, n)
# 	sol = solve_ivp(system(alpha, beta, gamma, delta, u), [t0, tf], np.array([x0, y0]), t_eval=t)
# 	return torch.from_numpy(sol.y).float()

# if __name__ == '__main__': 
# 	X = dataset(gamma=0.0)
# 	plt.figure(figsize=(8,8))
# 	plt.title('Unforced')
# 	plt.plot(X[0], X[1])


if __name__ == '__main__':
	vdp_experiment()


