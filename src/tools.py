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
		return (Tools.fee_constant*x1) - x0


	"""Returns the relative profit or rate of profit of
	the buying price of x0 and selling price of x1.

	:param x0: buying price
	:param x1: selling price
	:returns: rate of return
	"""
	@staticmethod
	def relative_return(x0, x1):
		return Tools.absolute_return(x0, x1) / x0


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


	@staticmethod
	def _handle_empty_listings(item, relative):
		value = np.nan
		if item['buys'] and item['sells']:
			low = item['buys'][0]['unit_price']
			high = item['sells'][0]['unit_price']
			value = Tools.relative_return(low, high) if relative else Tools.absolute_return(low, high)
		return value


	@staticmethod
	def _handle_file(data, json_data, relative):
		for item in json_data:
			ind = item['id']
			value = Tools._handle_empty_listings(item, relative)
			if not ind in data:	
				data[ind] = [value]
			else:
				data[ind].append(value)
		return data


	@staticmethod
	def list_of_returns(relative = True):
		data = {}
		datapath = '../data/'
		dirs = Tools.immidiate_subdirs(datapath)
		for d in dirs:
			f_path = datapath + d + '/snap.json'
			with open(f_path, 'r') as f:
				json_data = json.load(f)
				data = Tools._handle_file(data, json_data, relative)
		return data


	"""
	:returns: mean expected return
	"""
	@staticmethod
	def mean_returns(relative = True):
		returns = Tools.list_of_returns(relative)
		means = {}
		for key, lst in returns.items():
			#print(lst)
			means[key] = np.nanmean(lst)
		return means


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

# mean absolute return of assets 
# aret = [np.nanmean(r) for r in tools.list_of_returns(relative = False)]
# mean relative return of assets
# rret = [np.mean(r) for r in tools.list_of_returns(relative = True)]

# 1. Rewrite with numpy
# 2. change None to np.nan