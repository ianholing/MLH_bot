import csv
import pickle

'''
Para generar dos diccionarios: Uno para convertir
letra a indice y el otro para la conversion inversa.
'''

path_to_csv = '../mensajes_slack.csv' #path al csv generado con "get_slack_msg.py"

text = ''
with open(path_to_csv, 'r') as f:
	data_file = csv.reader(f)

	for row in data_file:
		text += row[2] + ''

# Texto en minuscula
#text = text.lower()

# Guardar el texto en un fichero. 
# Solo contiene los mensajes.
with open('text_dataset.txt', 'w') as f:
	f.write(text)

# Generar diccionarios con indices y caracter:
unique_chars = sorted(list(set(text)))
nchars = len(unique_chars)

char2int = dict((c,i) for i,c in enumerate(unique_chars))
int2char = dict((i,c) for i,c in enumerate(unique_chars))

# Guardar los diccionarios en un fichero:
with open('char2int.pkl', 'wb') as f:
	pickle.dump(char2int, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('int2char.pkl', 'wb') as f:
	pickle.dump(int2char, f, protocol=pickle.HIGHEST_PROTOCOL)
