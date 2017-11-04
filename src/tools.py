import os
import json
import numpy as np


class Tools:
	"""1) Posting a sell order for price x will cost 0.05*x
	   2) 0.1*x of the posted sell order will be paid as a tax
	   Thus only 0.85*x will be recieved from a succesfull sale
	"""
	fee_constant = 0.85


	"""Returns the absolute profit of buy price x0 and sell
	price x1, where fees have been taken into account

	:param x0: buying price
	:param x1: selling price
	:returns: absolute return of the prices
	"""
	@staticmethod
	def absolute_return(x0, x1):
		return (Tools.fee_constant*x1) -x0


	"""Returns the relative profit or rate of profit of
	the buying price of x0 and selling price of x1.

	:param x0: buying price
	:param x1: selling price
	:returns: rate of return
	"""
	@staticmethod
	def relative_return(x0, x1):
		return absolute_return(x0, x1) / x0


	"""Lists the immidiate subdirectories below the given path.

	:param root: path to be investigated
	:returns: a list of subdirectory names 
	"""
	@staticmethod
	def immidiate_subdirs(root):
		return [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]


	"""Check if a value is contained in a range
	"""
	@staticmethod
	def in_between(low, high, value):
		return low <= value and value <= high


	"""Returns the mean of the relative returns for a item with data_id i
	
	:param i: the data_id of the item
	:returns: the mean of the relative returns of the item asset
	"""
	@staticmethod
	def mean_relative_return(i):
		with open('../data/item_ids.json', 'r') as f:
			ids = json.load(f)
		try:
			index = ids.index(i)
		except ValueError:
			return None
		d = '../data/'
		files = os.listdir(d + Tools.immidiate_subdirs(d)[0])
		for filename in files:
			bounds = filename[:-5].split('-')
			low = int(bounds[0])
			high = int(bounds[1])
			if(Tools.in_between(low, high, index)): break
		#Continue
		return filename


	@staticmethod
	def print_price(currency):
		st = str(int(currency))
		if st[0] == '-':
			minus = st[0]
			st = st[1:]
		else:
			minus = ''
		copper = st[-2:]
		silver = '0' if st[-4:-2] == '' else st[-4:-2]
		gold = '0' if st[:-4] == '' else st[:-4]
		return minus + gold + " g " + silver + " s " + copper + " c"
