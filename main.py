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
			lr: float=0.01,
			momentum: float=0.9,
			n_epochs: int=1000,
			loss: Callable=None,
		):
		assert len(t_span) >= 2
		assert ndim > 0
		self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
		self.ndim = ndim
		self.dydt = dydt
		self.y0_bounds = y0_bounds
		self.batch_size = batch_size
		self.t_span = t_span
		self.t_span_torch = torch.from_numpy(t_span).to(self.device)
		self.model = model.double()
		self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
		self.n_epochs = n_epochs
		self.loss = nn.MSELoss() if loss is None else loss

	def get_batch(self):
		y0 = np.random.uniform(low=self.y0_bounds[:,0], high=self.y0_bounds[:,1], size=self.ndim)
		sol = solve_ivp(self.dydt, (self.t_span[0], self.t_span[-1]), y0, t_eval=self.t_span, method='LSODA')
		y0 = torch.from_numpy(y0).to(self.device)
		y = torch.from_numpy(sol.y.T).to(self.device)
		return y0, y

	def train_epoch(self):
		self.optimizer.zero_grad()
		y0, y = self.get_batch()
		pred_y = deq.odeint(self.model, y0, self.t_span_torch).to(self.device)
		loss = self.loss(pred_y, y)
		loss_val = loss.item()
		loss.backward()
		self.optimizer.step()
		return loss_val

	def run_and_visualize(self):
		loss_history = []
		for i in range(self.n_epochs):
			loss = self.train_epoch()
			print(f'Epoch: {i} Loss: {loss}')
			loss_history.append(loss)

		plt.plot(loss_history)
		plt.xlabel('Epoch')
		plt.ylabel('MSE Loss')
		plt.tight_layout()
		plt.show()


'''
Van der Pol oscillator
'''
def vdp_experiment():
	ndim = 2
	mu = 3.
	y0_bounds = np.array([[-1, 1], [-1, 1]])
	batch_size = 40
	t_span = np.linspace(0, 10, 500)

	def dydt(t, y):
		return np.array([y[1], mu*(1-y[0]**2)*y[1] - y[0]])

	class Model(nn.Module):
		def __init__(self):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(2, 20),
				nn.Tanh(),
				nn.Linear(20, 2),
			)
			for m in self.net.modules():
				if isinstance(m, nn.Linear):
					nn.init.normal_(m.weight, mean=0, std=0.1)
					nn.init.constant_(m.bias, val=0)

		def forward(self, t, y):
			return self.net(y)

	model = Model()
	experiment = NODESpectralExperiment(ndim, dydt, y0_bounds, batch_size, t_span, model)
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

