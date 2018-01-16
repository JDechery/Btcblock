# block chain transaction scrape
import requests
import pandas as pd

homeurl = 'https://blockchain.info/rawblock/'
block_hash_ex = '0000000000000000005048da3ecea6695cb217ebf85f1a03fcf11c006de078c7'

response = requests.get(homeurl+block_hash_ex)
blockdata = response.json()
prev_hash = blockdata['prev_block']
txfee = []  # (index, time, numin, numout, fee)
for tx in blockdata['tx']:
    temp = (tx[''])
