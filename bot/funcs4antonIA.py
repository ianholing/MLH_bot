import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle

class LSTMNN(nn.Module):
	def __init__(self, nchars, hid_s, n_layer):
		super(LSTMNN,self).__init__()

		self.nchars = nchars # Numero de caracteres.
		self.hid_s = hid_s # Neuronas en la capa oculta.
		self.n_layer = n_layer # Numero de capas recurrentes.

		self.encoder = nn.Embedding(self.nchars, self.hid_s)
		self.decoder = nn.Linear(self.hid_s, self.nchars)
		self.LSTM = nn.LSTM(self.hid_s, self.hid_s, self.n_layer)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, inp, hidden):
		batch = inp.size(0)
		y = self.encoder(inp).view(1, batch, -1)
		y, hidden = self.LSTM(y, hidden)
		y = self.decoder(y.view(batch,-1))
		y = self.softmax(y)
		return y, hidden

	def init_hidden(self,batch):
		h = Variable(torch.zeros(self.n_layer, batch, self.hid_s))
		return (h, h)


def load_antonIA(path):
	# Cargar estructura de la red, parametros y diccionarios char<->int.
	# path: directorio donde se encuentran almacenados los parámetros.
	data_dict = torch.load(path, map_location=torch.device('cpu'))

	# Generar modelo y cargar parametros:
	nchars, hs, nl = data_dict['structure']
	model = LSTMNN(nchars, hs, nl)
	model.load_state_dict(data_dict['state_dict'])
	model.eval()

	return model, data_dict['char2int'], data_dict['int2char']

def answer(model, char2int, int2char, msg):
	# Caracteres para terminar una frase:
	end_chars = ['.', '!', '?']
	end = []
	for char in end_chars:
		end.append(char2int[char])

	# Convertir el mensaje a un array de enteros.
	msg = msg.lower()
	inp = torch.zeros(1,len(msg)).long()
	for elem in range(len(msg)):
		inp[0,elem] = char2int[msg[elem]]

	inp = Variable(inp)

	# Inicializar a cero el hidden state.
	hidden = model.init_hidden(1)

	# Generar el hidden state pasando como input el mensaje (msg).
	for elem in range(len(msg) - 1):
		_, hidden = model(inp[:,elem], hidden)

	# Comenzar a generar la respuesta:
	inp = inp[0,[-1]]

	it = 0
	answer = ''
	while inp.item() not in end or it == 0:
		char_list, hidden = model(inp, hidden)
		_, next_char = char_list.max(1)

		answer += int2char[next_char.item()]

		if it > 300:
			# Por si no aparece un simbolo de terminar en mucho tiempo.
			answer += '... bueno, me estoy enrollando. ¡Un placer!'
			break

		inp = torch.zeros(1,1).long()
		inp[0,0] = next_char
		inp = Variable(inp)

		it += 1

	return answer
