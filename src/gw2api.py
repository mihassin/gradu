import urllib.request
import json

class Gw2Api:
	@staticmethod
	def getAllItemsList():
		return Gw2Api._request('items')
	
	''' Jos ids on tyhjä, palautetaan kaikkien itemien idt
	'''
	@staticmethod
	def listings(ids = []):
		tail = 'listings'
		if len(ids) != 0:
			tail += '?ids=' + ','.join(str(ind) for ind in ids)
		return Gw2Api._request('commerce', tail)

	@staticmethod
	def _url_length(url):
		return len(url) 

	@staticmethod
	def _request(*args):
		url = "http://api.guildwars2.com/v2/" + '/'.join(args)
		request = urllib.request.Request(url)
		return json.loads(urllib.request.urlopen(request).read())
