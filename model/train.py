import pickle
import numpy as np
import torch
from torch.autograd import Variable

from model import LSTMNN

import argparse

from generate import generate

parser = argparse.ArgumentParser(description='Text predictor')
parser.add_argument('--tp', type=int, default=100, help='Number of training points.')
parser.add_argument('--tl', type=int, default=20, help='Number of characters in each training point')
parser.add_argument('--mepoch', type=int, default=2000, help='Total number of training epochs.')
parser.add_argument('--hs', type=int, default=100, help='Hidden size.')
parser.add_argument('--nl', type=int, default=2, help='Number of recurrent layers.')
parser.add_argument('--cuda', action='store_true', help='Use cuda.')
args = parser.parse_args()

def read_text(file):
	with open(file, 'r') as f:
		text = f.read()
	return text, len(text)

def get_train_set(text, text_len, char2int):
	# Se generan Variables (de pytorch) que contendran el conjunto de 
	# entrenamiento.
	inp = torch.zeros([args.tp, args.tl]).long()
	out = torch.zeros([args.tp, args.tl]).long()

	for point in range(args.tp):
		# Se selecciona una parte random del texto:
		seed = np.random.randint(text_len - args.tl - 1)
		x = text[seed:seed+args.tl]

		# Almacenamos en y el mismo fragmento pero desplazado un caracter.
		y = text[seed+1:seed+args.tl+1]

		# Generamos los puntos de input y output:
		for elem in range(args.tl):
			inp[point, elem] = char2int[x[elem]]
			out[point, elem] = char2int[y[elem]]

	return Variable(inp), Variable(out)


train_set = 'text_dataset.txt'
text, text_len = read_text(train_set)

# Cargar diccionarios con conversion de caracteres a indices:
with open('char2int.pkl', 'rb') as f:
	char2int = pickle.load(f)

with open('int2char.pkl', 'rb') as f:
	int2char = pickle.load(f)

nchars = len(char2int.keys())

# Generar puntos input - output
inp, out = get_train_set(text, text_len, char2int)

# Definimos el modelo a entrenar:
model = LSTMNN(nchars, args.hs, args.nl)

# Definimos el algoritmo de minimizacion:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Definimos funcion de error:
loss_fn = torch.nn.CrossEntropyLoss()

if args.cuda:
	inp = inp.cuda()
	out = out.cuda()
	model = model.cuda()

# Empezar proceso de entrenamiento:
for epoch in range(args.mepoch):
	hidden = model.init_hidden(inp.size(0))
	if args.cuda:
		hidden = (hidden[0].cuda(), hidden[1].cuda())
	
	optimizer.zero_grad()
	loss = 0e0
	for elem in range(args.tl):
		y, hidden = model(inp[:,elem], hidden)
		loss += loss_fn(y, out[:,elem])

	loss.backward()
	optimizer.step()

	if epoch %10 == 0:
		generate(model, char2int, int2char, 100, cuda=args.cuda)
		print 'Epoch {}, loss {}'.format(epoch, loss.data.item())
		model.train()
