from matplotlib import pyplot as plt
import numpy as np
from math import sqrt


n_epch=4001
losses=[]
val_losses=[]
runs=[1,2,3,4,5]

for id in runs:
    path1=f'./results/losses/run {id}/Loss_Labeled Data_{n_epch}.txt'
    path2=f'./results/losses/run {id}/ValLoss_Labeled Data_{n_epch}.txt'

    loss=np.genfromtxt(path1,delimiter=',')
    val_loss=np.genfromtxt(path2,delimiter=',')

    losses.append(loss)
    val_losses.append(val_loss)

a,b,c,d,e=losses[0],losses[1],losses[2],losses[3],losses[4]
f,g,h,m,n=val_losses[0],val_losses[1],val_losses[2],val_losses[3],val_losses[4]


lst_std1=[]
for i in range(n_epch):
    st=np.std([a[i],b[i],c[i],d[i],e[i]])
    lst_std1.append(st)
lst_std1=np.array(lst_std1)


lst_std2=[]
for i in range(n_epch):
    st=np.std([f[i],g[i],h[i],m[i],n[i]])
    lst_std2.append(st)
lst_std2=np.array(lst_std2)


ci1=1.96*lst_std1/sqrt(5)
ci2=1.96*lst_std2/sqrt(5)

mu1=(losses[0]+losses[1]+losses[2]+losses[3]+losses[4])/5
mu2=(val_losses[0]+val_losses[1]+val_losses[2]+val_losses[3]+val_losses[4])/5

x1,x2=np.arange(n_epch),np.arange(n_epch)

plt.figure('Interval_plot',dpi=200)
plt.plot(x1,mu1,label='Train Data')
plt.plot(x2,mu2,label='Validation Data')
plt.fill_between( x1,(mu1-ci1), (mu1+ci1), color='blue', alpha=.2)
plt.fill_between( x2,(mu2-ci2), (mu2+ci2), color='red', alpha=.2)
plt.title('Labeled Data')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./Labeled Data_{n_epch}')
