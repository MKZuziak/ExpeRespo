import numpy as np
import utils
import itertools
from sklearn.linear_model import LogisticRegression

(X_train, y_train), (X_test, y_test) = utils.load_mnist()
print(len(X_train))
#print(X_train.shape)

train_set = [utils.partition(X_train, y_train, 100)[partition_id] for partition_id in range(100)]
test_set = [utils.partition(X_test, y_test, 100)[partition_id] for partition_id in range(100)]

X_centralised = []
y_centralised = []

#for cid_list in train_set:
    #for x in range(0, len(cid_list), 2):
        #print(len(cid_list))
        #for y in range(len(cid_list[x])):
            #train_set_centralised.append(cid_list[x][y])
            
            #print(cid_list[x][y])
            #print(cid_list[x+1][y])

for cid_list in train_set:
    for y in range(len(cid_list[0])):
        X_centralised.append(cid_list[0][y])
        y_centralised.append(cid_list[1][y])

test_set_centralised = test_set

central_model = LogisticRegression(
penalty="l2",
max_iter=1000,  # local epoch
warm_start=True,  # prevent refreshing weights when fitting
)

utils.set_initial_params(central_model)
central_model.fit(X_centralised, y_centralised)
central_model.score(test_set[1][0], test_set[1][1])