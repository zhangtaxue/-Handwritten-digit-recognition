import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image,ImageOps

model = keras.models.load_model('my_model.h5')     #载入模型

im = Image.open("7.png")  #读取图片路径
im = im.resize((28,28))   #调整大小和模型输入大小一致
im_grey = im.convert('L')    #将图片灰色化
im_inv = ImageOps.invert(im_grey)    #反色，变成黑底白字

#im_inv.show()
tp = np.array(im_inv)              #获取像素
tp[tp<=100]=0                      #极化，变成纯黑白
tp[tp>=140]=255


ret = model.predict((tp/255).reshape((1,28,28,1)))
number = np.argmax(ret)
print(number)
