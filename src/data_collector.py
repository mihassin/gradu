import os
import time
import json


from gw2api import Gw2Api as api

class DataCollector:


	def __init__(self):
		self.item_ids_path = '../data/item_ids.json'
		self.times_path = '../data/request_times.json'


	def _check_path(self, path, empty=True):
		if not os.path.exists(path):
			with open(path, 'w+') as f:
				data = [] if empty else api.listing()
				json.dump(data, f)


	def _housekeeping(self, listings_path):
		self._check_path(self.item_ids_path, False)
		with open(self.item_ids_path, 'r') as f:
			ids = json.load(f)
		os.mkdir(listings_path)
		return ids


	def _timekeeping(self, t):
		minutes = int(t/60)
		seconds = int(t - (minutes*60))
		print("Total time", minutes, "minutes", seconds, "seconds")
		self._check_path(self.times_path, [])
		with open(self.times_path, 'r') as f:
			times = json.load(f)
		with open(self.times_path, 'w') as f:
			times.append(t)
			json.dump(times, f)


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
