import os
import json
import time
import fileimport

try:
	from gw2api import Gw2Api as api
except ImportError:
	print("Couldn't locate gw2api.py")

item_ids_path = '../data/item_ids.json'
if not os.path.exists(item_ids_path):
	with open(item_ids_path, 'w+') as f:
		json.dump(api.listings(), f)

listings_path = '../data/' + time.strftime('%d-%m-%Y-%H:%M') + '.json'
if not os.path.exists(listings_path):
	with open(listings_path, 'w+') as f:
		json.dump([], f)

i = 0
j = 1
t = time.time()
while(i < len(ids)):
	print("Batch", j)
	if(i+200 > len(ids)):
		listings = api.listings(ids[i:len(ids)-1])
	else:
		listings = api.listings(ids[i:i+200])
	i += 200
	j += 1
print(time.time()-t)


''' TODO
def fix_structure(filename):
	with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
		for line in file:
			line.replace("\n][", ",")
'''