# seed the pseudorandom number generator
from random import seed
from random import randint
import json

# for _ in range(100):
# 	value = randint(0, 10000000)
# 	print(value)

dataList = []

with open('../../mini_sparkify_event_data.json') as f:

    for jsonObj in f:
        data = json.loads(jsonObj)
        dataList.append(data)

    print(len(dataList))

# seed random number generator
seed(1)
n_samples = len(dataList)
n_times = 10

for _ in range(n_times):

    value = randint(0, n_samples-1)
    # print(str(value%n_times))

    print(dataList[value]['page'])

    # data = json.load(json_file)
    # print(len(data))

    # for p in data:
    #     print('Name: ' + p['name'])
    #     print('Website: ' + p['website'])
    #     print('From: ' + p['from'])
    #     print('')    