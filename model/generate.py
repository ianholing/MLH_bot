import numpy as np
import torch
from torch.autograd import Variable

def generate(model, char_to_int, int_to_char, length, seed='a', cuda=False):
	print_text = seed
	inp = torch.zeros(1,len(seed)).long()

	for elem in range(len(seed)):
		inp[0,elem] = char_to_int[seed[elem]]

	with torch.no_grad():
		inp = Variable(inp)

	model.eval()

	hidden = model.init_hidden(1)

	if cuda:
		model = model.cuda()
		inp = inp.cuda()
		hidden = (hidden[0].cuda(), hidden[1].cuda())

	for elem in range(len(seed) - 1):
		_, hidden = model(inp[:,elem], hidden)

	# Begin prediction:
	inp = inp[:,[-1]]

	for elem in range(length):
		next_char_ind, hidden = model(inp[:,0], hidden)
		_, next_char = next_char_ind.max(1)
		print_text = print_text + int_to_char[next_char.item()]

		inp = torch.zeros(1,1).long()
		inp[0,0] = next_char
		inp = Variable(inp)
		if cuda:
			inp = inp.cuda()
		
	print(print_text)
