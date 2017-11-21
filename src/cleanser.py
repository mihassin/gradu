import json
import numpy as np

from tools import Tools as tools

class Cleanser:

	#TODO
	@staticmethod
	def trim_listings_and_missing_data():
		path = '../data/'
		subds = tools.immidiate_subdirs(path)
		files = tools.immidiate_subdirs(path+subds[0])
		for d in subds:
			trimmed = []
			for file in files:
				with open(path+d+'/'+file) as f:
					json_data = json.load(f)
				for item in json_data:
					value = {}
					if not item['buys'] or not item['sells']: continue
					value['id'] = item['id']
					value['buy'] = item['buys'][0]['unit_price']
					value['sell'] = item['sells'][0]['unit_price']
					trimmed.append(value)
			with open(path+)
json.dump()