import os
import time
import json


from gw2api import Gw2Api as api

class DataCollector:


	def __init__(self):
		self.item_ids_path = '../data/item_ids.json'


	def _check_ids(self):
		if not os.path.exists(self.item_ids_path):
			with open(item_ids_path, 'w+') as f:
				json.dump(api.listings(), f)


	def _housekeeping(self, listings_path):
		self._check_ids()
		with open(self.item_ids_path, 'r') as f:
			ids = json.load(f)
		os.mkdir(listings_path)
		return ids


	def _timekeeping(self, t):
		minutes = int(tt/60)
		seconds = int(tt - (minutes*60))
		print("Total time", minutes, "minutes", seconds, "seconds")


	def snapshot(self):
		listings_path = '../data/' + time.strftime('%d-%m-%Y-%H:%M')	
		ids = self._housekeeping(listings_path)
		i = 0
		j = 1
		t = time.time()
		while(i < len(ids)):
			print("Batch", j)
			print("Time so far", time.time()-t, "seconds")
			if(i+200 > len(ids)):
				k = len(ids)-1
				listings = api.listings(ids[i:k])
			else:
				k = i+200
				listings = api.listings(ids[i:k])
			fn = listings_path+"/"+str(i)+"-"+str(k)+".json"
			with open(fn, 'w+') as f:
				json.dump(listings, f)
			i += 200
			j += 1
		self._timekeeping(time.time() - t)
		


'''
def nasty_version(ids, path):
	i = 0
	j = 1
	t = time.time()
	while(i < len(ids)):
		print("Batch", j)
		if(i+200 > len(ids)):
			listings = api.listings(ids[i:len(ids)-1])
		else:
			listings = api.listings(ids[i:i+200])
		with open(path, 'r') as f:
			feed = json.load(f)
		feed.extend(listings)
		with open(path, 'w') as f:
			json.dump(feed, f)
		i += 200
		j += 1
	print(time.time()-t)
'''