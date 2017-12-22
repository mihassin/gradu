#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import datetime
import time
import json
import urllib.request

from gw2api import Gw2Api as api
from utils import get_project_root

class DataCollector:
	"""A Class responsible for the collection of data.
	DataCollector also informs user of the process since one
	round of snapshot() takes over 5 minutes
	"""


	def __init__(self):
		self.root = get_project_root()
		self.data_path = self.root + '/data'
		self.item_ids_path = self.data_path + '/item_ids.json'
		self.times_path = self.data_path + '/request_times.json'


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


	"""This helper function fetches the data

	:param i: index to keep track id of item list ids
	:param ids: list containing item ids
	:returns: tail index k and json data listings 
	"""
	def _fetch_batch(self, i, ids):
		if(i+200 > len(ids)):
			k = len(ids)-1
		else:
			k = i+200
		try:
			listings = api.listings(ids[i:k])
		except Exception as e:
			with open(self.root + '/logs/' + datetime.datetime.now().replace(microsecond=0).isoformat() + '.log', 'w+') as f:
				f.write(str(e))
				print('Failed to fetch data due to an exception:', e)
		return (k, listings)


	"""Takes a snapsnot of the GuildWars2 market by collection
	all the buy and sell listings of every item in the market.
	Due to some limitations of the gw2api only 200 items can be
	processed in a single request, which results in quite unelegant
	structure.
	"""
	def snapshot(self):
		listings_path = self.data_path + '/' + datetime.datetime.now().replace(microsecond=0).isoformat() + '/'	
		ids = self._housekeeping(listings_path)
		i = 0
		j = 1
		t = time.time()
		while(i < len(ids)):
			print("Batch", j, ": Time so far", time.time()-t, "seconds")
			#fetching data
			k, listings = self._fetch_batch(i, ids)
			fn = listings_path+str(i)+"-"+str(k)+".json"
			with open(fn, 'w+') as f:
				json.dump(listings, f)
			i += 200
			j += 1
		self._timekeeping(time.time() - t)
		return listings_path


	"""A function for clearing the data folder structure.
	All the separate json files within a date folder are 
	merged into a single json file snap.json.

	:param path: path to the date folder
	:returns: a success message
	"""
	def merge_snapshot(self, path):
		files = os.listdir(path)
		data = []
		for file in files:
			with open(path + file, 'r') as f:
				if not data:
					data = json.load(f)
				else:
					batch = json.load(f)
					data.extend(batch)
			os.remove(path+file)
		with open(path + 'snap.json', 'w+') as f:
			json.dump(data, f)
		return 'Merging complete!'


if __name__ == "__main__":
	dc = DataCollector()
	path = dc.snapshot()
	print('Merging batches')
	dc.merge_snapshot(path)

# TODO

'''
	1. if an exception occurs, loop the batch a few times | error handeling
'''