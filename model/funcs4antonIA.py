import torch
from torch.autograd import Variable

import pickle
from model import LSTMNN

def load_antonIA():
	# Cargar diccionarios de conversion int<->char
	with open('char2int.pkl', 'rb') as f:
		char2int = pickle.load(f)

	with open('int2char.pkl', 'rb') as f:
		int2char = pickle.load(f)

	# Generar modelo y cargar parametros:
	data_dict = torch.load('checkpoint.pth', map_location=torch.device('cpu'))

	nchars, hs, nl = data_dict['structure']
	model = LSTMNN(nchars, hs, nl)
	model.load_state_dict(data_dict['state_dict'])
	model.eval()

	return model, char2int, int2char

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
	for elem in xrange(len(msg) - 1):
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
			answer += '... bueno, me estoy enrollando. Un placer!'
			break

		inp = torch.zeros(1,1).long()
		inp[0,0] = next_char
		inp = Variable(inp)

		it += 1

	return answer