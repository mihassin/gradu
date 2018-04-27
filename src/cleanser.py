#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import json
import numpy as np


from utils import get_project_root
from utils import immidiate_subdirs


class Cleanser:
	'''Class for data cleansing
	'''


	'''Trims extra data from raw_data, only selects entries with
	required features and stores the trimmed ndarray into the data
	directory

	:param path: path in filesystem to the directory containing data file
	'''
	@staticmethod
	def trim_listings_and_missing_data(path):
		trimmed = np.array([['id' ,'buy', 'sell', 'demand', 'supply']])
		with open(path + '/snap.json', 'r') as f:
			raw_data = json.load(f)
		for obj in raw_data:
			if not obj['buys'] or not obj['sells']: continue
			row = [obj['id']]
			row.append(obj['buys'][0]['unit_price'])
			row.append(obj['sells'][0]['unit_price'])
			row.append(sum([o['quantity'] for o in obj['buys']]))
			row.append(sum([o['quantity'] for o in obj['sells']]))
			trimmed = np.append(trimmed, [row], axis=0)
		trimmed.dump(path+'/trimmed.ndarray')


	'''Loops through all data directories and prints status messages
	'''
	@staticmethod
	def clean_data():
		data_path = get_project_root() + '/data/'
		dirs = immidiate_subdirs(data_path)
		n = len(dirs)
		print('Preparing to clean the raw data..')
		Cleanser.status(0, n)
		for i, d in enumerate(dirs):
			dir_path = data_path+d
			if not os.path.exists(dir_path+'/trimmed.ndarray'):
				Cleanser.trim_listings_and_missing_data(dir_path)
			else:
				print(dir_path, 'has been already trimmed!')
			Cleanser.status(i+1, n)
		print('Cleaning done!')


	'''Prints a status message.

	:param i: index of current directory
	:param n: amount of directories
	'''
	@staticmethod
	def status(i, n):
		p = (i/n)*100
		print('Status:', i, "/", n, p, '% completed.')


	@staticmethod
	def create_R_matrix():
		data_path = get_project_root() + '/data'
		dirs = immidiate_subdirs(data_path)
		# remove directories with missing trimmed arrays
		for d in dirs:
			if not os.path.exists(data_path + '/' + d + '/trimmed.ndarray'):
				dirs.remove(d)
		headers = np.array(['date'])
		R = np.array([headers])
		return dirs


if __name__ == "__main__":
	Cleanser.clean_data()