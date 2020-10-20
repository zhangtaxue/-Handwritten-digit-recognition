import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

path = r'D:\编译器\python\mnist.npz'        #MNIST数据存储地址
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data(path)    #获取训练数据
x_train = x_train.reshape(x_train.shape[0],28,28,1)/255
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255

model = keras.models.Sequential()                                           #Sequential 用于搭建序列模型


model.add
#搭建卷积层和池化层
model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation = 'relu' ,input_shape=(28,28,1)))
#搭建第一层卷积层，卷积的数目为32 ，卷积窗的大小为3*3 ，激活函数用relu ，输入数据的模型为28*28*1
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#搭建第一层池化层，常用的池化层有MaxPlooling2D和AveragePooling2D，前者取最大值，后者取平均值，池化层大小设为2*2，步长为1
model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation = 'relu'))
#第二层卷积层
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#第二层池化层
#开始搭建全连接层
model.add(keras.layers.Flatten())
#Flatten 层用于展开张量
model.add(keras.layers.Dense(128,activation='relu'))
# Dense 层为全连接层，输出有 128 个神经元，激活函数使用 relu。激活函数用于实现去线性化。常见的还有sigmoid和tanh。
model.add(keras.layers.Dropout(0.25))
# Dropout 层使用 0.25 的失活率。随机的拿掉网络中的部分神经元，从而减小对W权重的依赖，以达到减小过拟合的效果。
model.add(keras.layers.Dense(10,activation='softmax'))
# 再接一个全连接层，激活函数使用 softmax，得到对各个类别预测的概率。softmax在分类问题中很常见

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
#设置优化方法为adam，损失函数为sparse_categorical_crossentropy，评价函数为accuracy
model.fit(x_train,y_train,epochs = 5)
#fit用于训练模型，epochs为训练轮数，这里设置5
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
#evaluate用于评估模型，返回的数值分别是损失和指标。

model.save('my_model.h5')  #保存模型

#随机抽取一个数测试
i = np.random.random_integers(0,len(x_test))
plt.imshow(x_test[i],cmap=plt.cm.binary)
plt.show()

predictions = model.predict(x_test)
print(np.argmax(predictions[i]))
