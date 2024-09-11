from sklearn.metrics import confusion_matrix
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import PlotHistory
from sklearn.metrics import auc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Activation,BatchNormalization,Dense,Flatten,\
                         Reshape,Dropout,Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import cv2


def ROC_Diagram_Single(x,counter,actual,dis_v_sup):

    ns_probs = [0 for _ in range(len(actual))]
    lr_probs = dis_v_sup.predict_proba(x)
    lr_probs = lr_probs[:, 1]
    ns_fpr, ns_tpr, _ = roc_curve(actual, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(actual, lr_probs)

    plt.figure(f'ROC Diagram_{counter}',dpi=200)
    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.',label=f'{counter} Iteration')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'./results/ROC Diagram{counter}_single.png') 




def predict_period(counter,training_time,dis_v_sup):
    x, y = dataset.read_testingdata()
    y = to_categorical(y,num_classes=num_classes)

    test_loss, test_acc = dis_v_sup.evaluate(x,y)
    dis_v.save(f'./weights/dis_v_{counter}.h5')
    dis_v_sup.save(f'./weights/dis_v_sup_{counter}.h5')
    dis_v_unsup.save(f'./weights/dis_v_unsup_{counter}.h5')
    gen_v.save(f'./weights/gen_v_{counter}.h5')

    # confusion matrix
    test_label_p=dis_v_sup.predict(x)
    test_label_p=np.argmax(test_label_p,axis=1)
    y0=np.argmax(y,axis=1)
    actual=list(y0)
    predicted=list(test_label_p)
    c=confusion_matrix(actual,predicted,labels=[0,1])

    ROC_Diagram_Single(x,counter,actual,dis_v_sup)

    ns_probs = [0 for _ in range(len(actual))]

    lr_probs = dis_v_sup.predict_proba(x)
    lr_probs = lr_probs[:, 1]

    ns_fpr, ns_tpr, _ = roc_curve(actual, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(actual, lr_probs)

    plt.figure('ROC Diagram',dpi=200)
    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.',label=f'{counter} Iteration')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    if counter>=4500:
        plt.savefig(f'./results/ROC Diagram_All')
    a=auc(lr_fpr,lr_tpr)

 
    r=classification_report(actual,predicted)

    results_path=f'./results/results_{counter}.txt' 
    f1=open(results_path,'a')
    f1.write('Accuracy: '+str(test_acc)+'\n'+'Loss: '+str(test_loss)+'\n'+'AUC: '+str(a)+'\n')
    f1.write('\n\nMetrics for GAN: \n\n')
    f1.write('\n\n'+str(r))
    f1.write('\n\nTraining Time: '+str(training_time))
    f1.write('\n\nCofusion Matrix: \n'+str(c))
    f1.close()




def PreparingData():

    #read normals files
    normals=[]
    main_path='./Public Dataset 9/Normal-2/Normal/'
    main_folders=next(os.walk(main_path))[1]
    for i in main_folders:
        path=main_path+i+'/'
        folders=next(os.walk(path))[1]
        for x in folders:
            new_path=path+x+'/'
            data=glob.glob(new_path+'*.jpg')+glob.glob(new_path+'*.jpeg')+glob.glob(new_path+'*.png')
            if len(data)<1:
                indent_folders=next(os.walk(new_path))[1]
                for y in indent_folders:
                    new_path=new_path+y+'/'
                    data=glob.glob(new_path+'*.jpg')+glob.glob(new_path+'*.jpeg')+glob.glob(new_path+'*.png')
            normals.extend(data)


    #read sicks files
    sicks=[]
    main_path='./Public Dataset 9/COV-1/COV-1/NCP/'
    main_folders=next(os.walk(main_path))[1]
    for i in main_folders:
        path=main_path+i+'/'
        folders=next(os.walk(path))[1]
        for x in folders:
            new_path=path+x+'/'
            data=glob.glob(new_path+'*.jpg')+glob.glob(new_path+'*.jpeg')+glob.glob(new_path+'*.png')
            if len(data)<1:
                indent_folders=next(os.walk(new_path))[1]
                for y in indent_folders:
                    new_path=new_path+y+'/'
                    data=glob.glob(new_path+'*.jpg')+glob.glob(new_path+'*.jpeg')+glob.glob(new_path+'*.png')
            sicks.extend(data)


    #load normal files
    labels_n=[]
    train_data_n=[]
    for id in normals:    
        img=cv2.imread(id)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        img=cv2.resize(img,(100,100))
        img=img.astype('float32')
        train_data_n.append(img)
        labels_n.append(0)

   
    #load sick files
    labels_s=[]
    train_data_s=[]
    for id in sicks:    
        img=cv2.imread(id)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        img=cv2.resize(img,(100,100))
        img=img.astype('float32')
        train_data_s.append(img)
        labels_s.append(1)

    train_data_n.extend(train_data_s)
    labels_n.extend(labels_s)

    x=np.array(train_data_n)
    y=np.array(labels_n)

    x_train,x_test,train_labels,test_labels=train_test_split(x,y,test_size=0.2,random_state=0)
    print('train data:',len(x_train),'train labels:',len(train_labels),'test data:',len(x_test),'test labels:',len(test_labels))
    return (x_train,train_labels),(x_test,test_labels)


class Dataset:
    def __init__(self,num_labeled):

        self.num_labeled=num_labeled
        (self.x_train, self.y_train), (self.x_test, self.y_test)=PreparingData()
        

        def preprocess_img(x):
            x = (x.astype(np.float)-127.5) / 127.5
            x = np.expand_dims(x,axis=3)
            return x

        def preprocess_label(y):
            return y.reshape(-1,1)

        self.x_train=preprocess_img(self.x_train)
        self.y_train=preprocess_label(self.y_train)
        self.x_test=preprocess_img(self.x_test)
        self.y_test=preprocess_label(self.y_test)

    def read_batch_labeled(self,batch_size):
        ids = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[ids]
        labels = self.y_train[ids]
        return imgs,labels

    def read_batch_unlabeled(self,batch_size):
        ids = np.random.randint(self.num_labeled,self.x_train.shape[0], batch_size)
        imgs = self.x_train[ids]
        return imgs

    def read_trainingdata(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train,y_train

    def read_testingdata(self):
        return self.x_test,self.y_test


num_labeled=800
img_rows=100
img_cols=100
channels=1

dataset = Dataset(num_labeled)

img_shape = (img_rows,img_cols,channels)

zdim=100
num_classes = 2

def build_gen(zdim):
    model = Sequential()
    model.add(Dense(256*25*25,input_dim=zdim))
    model.add(Reshape((25,25,256)))
    model.add(Conv2DTranspose(64,kernel_size=3,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(32,kernel_size=3,strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'))
    model.add(Activation('tanh'))
    return model

def build_dis(img_shape):
    model=Sequential()
    model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=img_shape,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64,kernel_size=3,strides=2,input_shape=img_shape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128,kernel_size=3,strides=2,input_shape=img_shape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes))
    return model

def build_dis_supervised(dis_net):
    model = Sequential()
    model.add(dis_net)
    model.add(Activation('sigmoid'))
    return model

def build_dis_unsupervised(dis_net):
    model = Sequential()
    model.add(dis_net)

    def predict(x):
        prediction = 1.0 - (1.0/(K.sum(K.exp(x),axis=-1,keepdims=True)+1.0))
        return prediction

    model.add(Lambda(predict))
    return model

def build_gan(gen,dis):
    model = Sequential()
    model.add(gen)
    model.add(dis)
    return model

dis_v = build_dis(img_shape)

dis_v_sup = build_dis_supervised(dis_v)
dis_v_sup.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

dis_v_unsup = build_dis_unsupervised(dis_v)
dis_v_unsup.compile(loss='binary_crossentropy',optimizer=Adam())

gen_v = build_gen(zdim)
dis_v_unsup.trainable=False
gan_v = build_gan(gen_v,dis_v_unsup)
gan_v.compile(loss='binary_crossentropy',optimizer=Adam())

supervised_losses=[]
iteration_checks=[]

def train(iterations,batch_size,interval):
    import datetime
    start=datetime.datetime.now()  

    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size, 1))
    acc=[]
    acc2=[]
    acc3=[]
    acc4=[]
    acc5=[]
    acc6=[]
    acc7=[]
    acc8=[]
    for iteration in range(iterations):

        imgs, labels = dataset.read_batch_labeled(batch_size)
        labels = to_categorical(labels,num_classes=num_classes)

        imgs_unlabeled = dataset.read_batch_unlabeled(batch_size)

        z=np.random.normal(0,1,(batch_size,100))

        #1
        x,x2,y,y2=train_test_split(imgs,labels,test_size=0.2,random_state=0)
        dloss_sup, accuracy = dis_v_sup.train_on_batch(x,y)
        dloss_sup2, accuracy2 = dis_v_sup.train_on_batch(x2,y2)
        acc.append(accuracy)
        acc2.append(accuracy2)
        acc7.append(dloss_sup)
        acc8.append(dloss_sup2)

        #2
        x3,x4,y3,y4=train_test_split(imgs_unlabeled,real,test_size=0.2,random_state=0)
        dloss_real= dis_v_unsup.train_on_batch(x3,y3)
        accuracy3=dis_v_unsup.evaluate(x3,y3)
        acc3.append(accuracy3)
        accuracy4=dis_v_unsup.evaluate(x4,y4)
        acc4.append(accuracy4)

        #3
        x5,x6,y5,y6=train_test_split(imgs_unlabeled,fake,test_size=0.2,random_state=0)
        dloss_fake = dis_v_unsup.train_on_batch(x5,y5)
        accuracy5=dis_v_unsup.evaluate(x5,y5)
        acc5.append(accuracy5)
        accuracy6=dis_v_unsup.evaluate(x6,y6)
        acc6.append(accuracy6)


        dloss_unsup = 0.5 * np.add(dloss_real,dloss_fake)

        z = np.random.normal(0, 1, (batch_size, 100))

        if (iteration+1) % interval == 0:
            supervised_losses.append(dloss_sup)
            iteration_checks.append(iteration+1)

            print("%d [D loss supervised: %.4f , acc: %.2f] [D loss unsupervised: %.4f]" %
                  (iteration+1,dloss_sup,100.0*accuracy,dloss_unsup))

        if iteration==3500 or iteration==4000 or iteration==4500:
            end=datetime.datetime.now()
            training_time=end-start
            predict_period(iteration,training_time,dis_v_sup)
            PlotHistory.NetPlot(acc,acc2,acc3,acc4,acc5,acc6,acc7,acc8)


train(4501,32,800)







