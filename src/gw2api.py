import urllib.request
import json

class Gw2Api:
	'''
		If ids is empty, a list of valid ids is returned
	'''
	@staticmethod
	def listings(ids = []):
		tail = 'listings'
		if len(ids) != 0:
			tail += '?ids=' + ','.join(str(ind) for ind in ids)
		return Gw2Api._request('commerce', tail)

	'''
		A function that builds requests from *args to
		https://api.guildwars2.com

		version 2 of the api is used
	'''
	@staticmethod
	def _request(*args):
		url = "http://api.guildwars2.com/v2/" + '/'.join(args)
		request = urllib.request.Request(url)
		return json.loads(urllib.request.urlopen(request).read())
