# -*- coding: utf-8 -*-
import pdb
import random
import numpy as np
from more_itertools import locate

np.random.seed(44)
permutation = np.random.permutation(50)
file_reading = 'revisited_imagenet_2012_val.csv'
with open(file_reading, 'r') as data:
    ctnts = data.readlines()

# First for loop. Getting all the classes in an ordered manner.
classes = []
for line in ctnts:
    name, class_id = line.strip().split(',')
    classes.append(class_id)

writing_list = []
for class_id in range(1000):
    class_elems = list(locate(classes, lambda x: x == str(class_id)))
    retrieving = [class_elems[int(permutation[x])] for x in range(5)]
    from_orign = [ctnts[idx] for idx in retrieving]
    writing_list+=from_orign

np.random.shuffle(writing_list)
with open('rand5k_5perclass_val.csv', 'w') as data:
    for line in writing_list:
        data.write(line)
