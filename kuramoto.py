import os
import argparse
import time
import numpy as np
import pdb

import torch
import torch.fft
import torch.nn as nn
import torch.optim as optim
import scipy.fftpack

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=200)
parser.add_argument('--batch_time', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--niters', type=int, default=3000)
parser.add_argument('--test_freq', type=int, default=25)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
	from torchdiffeq import odeint_adjoint as odeint
else:
	from torchdiffeq import odeint

np.random.seed(1000)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

''' Van der Pol '''
# true_y0 = torch.tensor([[1., 0.]]).to(device)
# t = torch.linspace(0., 15., args.data_size).to(device)

''' Duffing ''' 
# true_y0 = torch.tensor([[-1., 2.]]).to(device)
# t = torch.linspace(0., 35., args.data_size).to(device)

''' Lorenz ''' 
# true_y0 = torch.tensor([[0, 1, 1.05]]).to(device)
# t = torch.linspace(0., 100., args.data_size).to(device)

''' Chua ''' 
# true_y0 = torch.tensor([[1.0, 0, 0]]).to(device)
# t = torch.linspace(0., 15., args.data_size).to(device)

''' Kuramoto ''' 
N = 3
true_y0 = torch.tensor([np.random.uniform(size=N).tolist()]).to(device)
t = torch.linspace(0., 3., args.data_size).to(device)
n_show = args.data_size // 2
omegas = torch.linspace(50, 75, N)


class Lambda(nn.Module):
	''' Van der Pol '''
	# mu = 3.

	# def forward(self, t, y):
	# 	y0, y1 = y[0][0], y[0][1]
	# 	return torch.Tensor([y1, self.mu*(1-y0**2)*y1 - y0])

	''' Duffing ''' 
	# alpha = -1.2
	# beta = 1.2
	# delta = 0.2

	# def forward(self, t, y):
	# 	y0, y1 = y[0][0], y[0][1]
	# 	return torch.Tensor([y1, -self.delta*y1 - self.alpha*y0 - self.beta*(y0**3)])

	''' Lorenz '''
	# sigma = 10
	# beta = 2.667
	# rho = 28

	# def forward(self, t, y):
	# 	y0, y1, y2 = y[0][0], y[0][1], y[0][2]
	# 	return torch.Tensor([
	# 		-self.sigma*(y0-y1),
	# 		self.rho*y0 - y1 - y0*y2,
	# 		-self.beta*y2 + y1*y2
	# 	])

	''' Chua '''
	# alpha = 15.395
	# beta = 28
	# R = -1.143
	# C_2 = -0.714

	# def forward(self, t, y):
	# 	x, y, z = y[0][0], y[0][1], y[0][2]
	# 	f_x = self.C_2*x + 0.5*(self.R-self.C_2)*2*torch.tanh(4*x)
	# 	return torch.Tensor([self.alpha*(y-x-f_x), x - y + z, -self.beta * y])

	''' Kuramoto '''
	K = 2.

	def forward(self, t, y):
		Y = torch.diag(y[0])
		M = torch.ones((N,N))
		Z = (M @ Y) - (Y @ M)
		Z = torch.sin(Z)
		z = Z.sum(axis=1) * self.K / N
		return omegas + z

def obs(y):
	return torch.sin(y)


with torch.no_grad():
	true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
	s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
	batch_y0 = true_y[s]  # (M, D)
	batch_t = t[:args.batch_time]  # (T)
	batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
	return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)


if args.viz:
	makedirs('png')
	import matplotlib.pyplot as plt
	from matplotlib.colors import LogNorm
	from pylab import figure, cm
	fig = plt.figure(figsize=(16, 6), facecolor='white')
	ax_traj = fig.add_subplot(121, frameon=False)
	# ax_phase = fig.add_subplot(132, frameon=False)
	# ax_vecfield = fig.add_subplot(143, frameon=False)
	ax_spectra = fig.add_subplot(122, frameon=False)
	plt.show(block=False)

# cb = None


def visualize(true_y, pred_y, odefunc, itr, spectra, freqs, epochs):

	if args.viz:
		# global cb
		''' Van der Pol '''
		# xmin, xmax = -2, 2
		# ymin, ymax = -4, 4

		''' Duffing ''' 
		# xmin, xmax = -2, 2
		# ymin, ymax = -2, 2

		''' Lorenz ''' 
		# xmin, xmax = -20, 20
		# ymin, ymax = -20, 20

		''' Chua ''' 
		# xmin, xmax = -3.0, 3.0
		# ymin, ymax = -0.8, 0.8
		# traj_min, traj_max = -4., 4.

		''' Kuramoto '''
		xmin, xmax = -1., 1.
		ymin, ymax = -1., 1.
		traj_min, traj_max = -1., 1.

		ax_traj.cla()
		ax_traj.set_title('Trajectories')
		ax_traj.set_xlabel('t')
		ax_traj.set_ylabel('x,y')
		for k in range(true_y.shape[2]):
			ax_traj.plot(t.cpu().numpy()[:n_show], obs(true_y).cpu().numpy()[:n_show, 0, k], 'g-', label='true' if k == 0 else '')
			ax_traj.plot(t.cpu().numpy()[:n_show], obs(pred_y).cpu().numpy()[:n_show, 0, k], 'b--', label='predicted' if k == 0 else '')
		# ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
		# ax_traj.set_ylim(traj_min, traj_max)
		ax_traj.legend()

		# ax_phase.cla()
		# ax_phase.set_title('Phase Portrait')
		# ax_phase.set_xlabel('x')
		# ax_phase.set_ylabel('y')
		# ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
		# ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
		# ax_phase.set_xlim(xmin, xmax)
		# ax_phase.set_ylim(ymin, ymax)

		# ax_vecfield.cla()
		# ax_vecfield.set_title('Learned Vector Field')
		# ax_vecfield.set_xlabel('x')
		# ax_vecfield.set_ylabel('y')

		# y, x = np.mgrid[ymin:ymax:21j, xmin:xmax:21j]
		# dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
		# mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
		# dydt = (dydt / mag)
		# dydt = dydt.reshape(21, 21, 2)

		# ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
		# ax_vecfield.set_xlim(xmin, xmax)
		# ax_vecfield.set_ylim(ymin, ymax)

		spectra = np.flip(np.array(spectra).T, axis=0)
		bin_nums = np.flip(np.arange(freqs.size), axis=0)
		ax_spectra.cla()
		ax_spectra.set_title('DCT Relative Error')
		ax_spectra.set_xlabel('Epoch')
		ax_spectra.set_ylabel('Frequency bins')
		im = ax_spectra.imshow(spectra, norm=LogNorm(vmin=spectra.min(), vmax=spectra.max()), interpolation='none', extent=[min(epochs), max(epochs), min(bin_nums), max(bin_nums)])
		ax_spectra.set_aspect('auto')
		# if cb != None: 
		# 	cb.remove()
		# cb = fig.colorbar(im, ax=ax_spectra, orientation='horizontal', pad=0.05)

		fig.tight_layout()
		plt.savefig('png/{:03d}'.format(itr))
		plt.draw()
		plt.pause(0.001)


class ODEFunc(nn.Module):

	def __init__(self):
		super(ODEFunc, self).__init__()

		self.fc1 = nn.Linear(N, 50)
		self.s1 = nn.Tanh()
		self.fc2 = nn.Linear(50, N)

		fcs = [self.fc1, self.fc2]
		for m in fcs:
			nn.init.normal_(m.weight, mean=0, std=0.1)
			nn.init.constant_(m.bias, val=0)

	def forward(self, t, y):
		z = self.fc1(y)
		z = self.s1(z)
		z = self.fc2(z)
		return z


class RunningAverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, momentum=0.99):
		self.momentum = momentum
		self.reset()

	def reset(self):
		self.val = None
		self.avg = 0

	def update(self, val):
		if self.val is None:
			self.avg = val
		else:
			self.avg = self.avg * self.momentum + val * (1 - self.momentum)
		self.val = val


if __name__ == '__main__':

	ii = 0

	func = ODEFunc().to(device)
	
	optimizer = optim.Adam(func.parameters(), lr=1e-3)
	end = time.time()

	time_meter = RunningAverageMeter(0.97)
	
	loss_meter = RunningAverageMeter(0.97)

	epochs = []
	spectra = []
	# freqs = np.arange(args.data_size) * 2 * np.pi / args.data_size # FFT frequencies, lo to hi
	freqs = np.arange(args.data_size) / args.data_size # FFT frequencies, lo to hi
	bins = 120
	bin_cutoff = 60
	freq_bins = np.digitize(freqs, np.linspace(freqs.min(), freqs.max(), bins+1), right=True)
	freq_bins[0] = 1
	freqs = np.array([freqs[freq_bins == i].max() for i in range(1, bin_cutoff+1)])
	spec_fun = lambda sig: scipy.fftpack.dct(sig.cpu().numpy())
	# spec_fun = lambda sig: scipy.fftpack.fft(sig.cpu().numpy())
	# spec_fun = lambda sig: torch.fft.fft(sig).cpu().numpy()
	# pdb.set_trace()

	try:
		for itr in range(1, args.niters + 1):
			optimizer.zero_grad()
			batch_y0, batch_t, batch_y = get_batch()
			pred_y = odeint(func, batch_y0, batch_t).to(device)

			loss = torch.mean(torch.abs(pred_y - batch_y))
			loss.backward()
			optimizer.step()

			time_meter.update(time.time() - end)
			loss_meter.update(loss.item())

			if itr % args.test_freq == 0:
				with torch.no_grad():
					pred_y = odeint(func, true_y0, t)

					if itr // args.test_freq > 0:
						# Store spectra
						loss_sig = torch.abs(pred_y - true_y)[:,0,0]
						# spectrum = spec_fun(loss_sig)
						axis = 1
						baseline = spec_fun(obs(true_y[:,0,axis]))
						spectrum = np.abs(spec_fun(obs(pred_y[:,0,axis])) - baseline) / np.abs(baseline)
						# pdb.set_trace()
						spectrum = np.array([spectrum[freq_bins == i].sum() for i in range(1, bin_cutoff+1)])
						# pdb.set_trace()
						spectra.append(spectrum)
						epochs.append(itr)
						visualize(true_y, pred_y, func, ii, spectra, freqs, epochs)

					loss = torch.mean(torch.abs(pred_y - true_y))
					print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
					ii += 1


			end = time.time()
	except KeyboardInterrupt:
		print('saving')
		plt.savefig('figure.pdf')
		plt.savefig('figure.png')
		raise KeyboardInterrupt

		# Visualize frequency of loss

