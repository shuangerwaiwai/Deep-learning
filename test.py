import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 25
plt.imshow(train_set_x_orig[index])

print("y=" + str(train_set_y[:,index]) + ", it is a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture")

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

# print("number_train: m_train = " + str(m_train))
# print("number_test: m_test = " + str(m_test))
# print("weight/high of picture: num_px = " + str(num_px))
# print("size of picture: " + str(num_px) + "," + str(num_px) + ", 3)")
# print("dim of train picture: " + str(train_set_x_orig.shape))
# print("dim of train tag: " + str(train_set_y.shape))
# print("dim of test picture: " + str(test_set_x_orig.shape))
# print("dim of test tag: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# print("训练集降维后的维度： " + str(train_set_x_flatten.shape))
# print("训练集标签的维度： "+ str(train_set_y.shape))
# print("测试集降维后的维度： " + str(test_set_x_flatten.shape))
# print("测试集标签的维度： " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmod(z):
    s = 1/(1+np.exp(-z))
    return s
train_set_x_orig[index]

#print(str(sigmod(9.2)))



