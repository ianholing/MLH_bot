import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMNN(nn.Module):
	def __init__(self, nchars, hid_s, n_layer):
		super(LSTMNN,self).__init__()

		self.nchars = nchars # Numero de caracteres.
		self.hid_s = hid_s # Neuronas en la capa oculta.
		self.n_layer = n_layer # Numero de capas recurrentes.

		self.encoder = nn.Embedding(self.nchars, self.hid_s)
		self.decoder = nn.Linear(self.hid_s, self.nchars)
		self.LSTM = nn.LSTM(self.hid_s, self.hid_s, self.n_layer)

	def forward(self, inp, hidden):
		batch = inp.size(0)
		y = self.encoder(inp).view(1, batch, -1)
		y, hidden = self.LSTM(y, hidden)
		y = self.decoder(y.view(batch,-1))
		return y, hidden

	def init_hidden(self,batch):
		h = Variable(torch.zeros(self.n_layer, batch, self.hid_s))
		return (h, h)
