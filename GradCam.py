import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
from tensorflow.compat.v1 import keras
from vis.utils import utils
from vis.visualization import visualize_cam

model=load_model('./weights/dis_v_4000.h5')
img0=cv2.imread('./pics/After Sobel Filtering/Sick 1.jpg')
img0=cv2.resize(img0,(100,100))
img=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
img=np.reshape(img,(100,100,1))

for ilayer,layer in enumerate(model.layers):
    print("{:3.0f} {:10}".format(ilayer,layer.name))


y_pred=model.predict(img[np.newaxis,...])
clas_idx_sorted=np.argsort(y_pred.flatten())[::-1]
topNclasses=5
for i,idx in enumerate(clas_idx_sorted[:topNclasses]):
    print("Top {} predict classs:    [index={}])={:5.3f}".format(i+i,idx,y_pred[0,idx]))


layer_idx=utils.find_layer_idx(model,'dense_1')
model.layers[layer_idx].activation=keras.activations.sigmoid


ixs = [8,9,10]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)


penultimate_layer_idx=utils.find_layer_idx(model,'conv2d_3')
class_idx=clas_idx_sorted[0]
seed_input=img
grad_top1=visualize_cam(model,layer_idx,class_idx,seed_input,penultimate_layer_idx=penultimate_layer_idx,backprop_modifier=None,grad_modifier=None)


def plot_map(grads):
    plt.figure('GradCam',dpi=200)
    plt.imshow(img0)
    i=plt.imshow(grads,cmap='jet',alpha=0.8)
    plt.colorbar(i)
    plt.show()


plot_map(grad_top1)










