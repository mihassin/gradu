import os
import time
import json
import urllib.request

from gw2api import Gw2Api as api

class DataCollector:
	"""A Class responsible for the collection of data.
	DataCollector also informs user of the process since one
	round of snapshot() takes over 5 minutes
	"""


	def __init__(self):
		self.item_ids_path = '../data/item_ids.json'
		self.times_path = '../data/request_times.json'


	"""Checks if a given path(file) exists.
	If empty is true, then a empty list is stored to path.
	Otherwise data from api.listings() is stored to path.

	:param path: location to be checked
	:param empty: controlls the type of data that is stored
	"""
	def _check_path(self, path, empty=True):
		if not os.path.exists(path):
			with open(path, 'w+') as f:
				data = [] if empty else api.listing()
				json.dump(data, f)


	"""Makes sure that item ids are available.
	Creates a directory for a new snapshot.
	:param listings_path: path for the snapshot
	:returns: list of item ids
	"""
	def _housekeeping(self, listings_path):
		self._check_path(self.item_ids_path, False)
		with open(self.item_ids_path, 'r') as f:
			ids = json.load(f)
		os.mkdir(listings_path)
		return ids


	"""Reports the time used by snapshot:s looping procedure.
	Also the reported time is stored to further analysis

	:param t: tells the used time in snapshot:s loop
	"""
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

	"""Takes a snapsnot of the GuildWars2 market by collection
	all the buy and sell listings of every item in the market.
	Due to some limitations of the gw2api only 200 items can be
	processed in a single request, which results in quite unelegant
	structure.
	"""
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
		

if __name__ == "__main__":
	dc = DataCollector()
	try:
		dc.snapshot()
	except urllib.error.HTTPError as err:
		with open('../logs/'+time.strftime('%d-%m-%Y-%H:%M')+'.log', 'w+') as f:
			f.write(str(err.code))
			print('Exiting due to Http error')