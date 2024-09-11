import numpy as np
def NetPlot(acc,acc2,acc3,acc4,acc5,acc6,acc7,acc8):
    import matplotlib.pyplot as plt
    
    counter=len(acc)
   
    plt.figure('Accuracy Diagram1',dpi=200)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(acc,color='blue')
    plt.plot(acc2,color='orange')
    plt.title('Labeled Data')
    plt.legend(['Train Data','Validation Data'])       
    plt.savefig(f'./Accuracy Diagram1_{counter}')


    results_path=f'./Acc_Labeled Data_{counter}.txt' 
    f1=open(results_path,'a')
    s1=str(acc)
    s1=s1.replace(']','')
    s1=s1.replace('[','')
    f1.write(s1)
    f1.close()
    results_path=f'./ValAcc_Labeled Data_{counter}.txt' 
    f2=open(results_path,'a')
    s2=str(acc2)
    s2=s2.replace(']','')
    s2=s2.replace('[','')
    f2.write(s2)
    f2.close()



    plt.figure('Loss Diagram1',dpi=200)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(acc3,color='blue')
    plt.plot(acc4,color='orange')
    plt.title('Unlabeled Data (Real)')
    plt.legend(['Train Data','Validation Data'])       
    # plt.show()
    plt.savefig(f'./results/Loss Diagram1_{counter}')


    results_path=f'./results/losses/Loss_Unlabeled Data (Real)_{counter}.txt' 
    f3=open(results_path,'a')
    s3=str(acc3)
    s3=s3.replace(']','')
    s3=s3.replace('[','')  
    f3.write(s3)
    f3.close()
    results_path=f'./results/losses/ValLoss_Unlabeled Data (Real)_{counter}.txt' 
    f4=open(results_path,'a')
    s4=str(acc4)
    s4=s4.replace(']','')
    s4=s4.replace('[','')
    f4.write(s4)
    f4.close()


    plt.figure('Loss Diagram2',dpi=200)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(acc5,color='blue')
    plt.plot(acc6,color='orange')
    plt.title('Unlabeled Data (Fake)')
    plt.legend(['Train Data','Validation Data'])    
    # plt.show()
    plt.savefig(f'./results/Loss Diagram2_{counter}')


    results_path=f'./results/losses/Loss_Unlabeled Data (Fake)_{counter}.txt' 
    f5=open(results_path,'a')
    s5=str(acc5)
    s5=s5.replace(']','')
    s5=s5.replace('[','')  
    f5.write(s5)
    f5.close()
    results_path=f'./results/losses/ValLoss_Unlabeled Data (Fake)_{counter}.txt' 
    f6=open(results_path,'a')
    s6=str(acc6)
    s6=s6.replace(']','')
    s6=s6.replace('[','')  
    f6.write(s6)
    f6.close()

    plt.figure('Loss Diagram3',dpi=200)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(acc7,color='blue')
    plt.plot(acc8,color='orange')
    plt.title('Labeled Data')
    plt.legend(['Train Data','Validation Data'])       
    plt.savefig(f'./Loss Diagram3_{counter}')


    results_path=f'./results/losses/Loss_Labeled Data_{counter}.txt' 
    f7=open(results_path,'a')
    s7=str(acc7)
    s7=s7.replace(']','')
    s7=s7.replace('[','')  
    f7.write(s7)
    f7.close()
    results_path=f'./results/losses/ValLoss_Labeled Data_{counter}.txt' 
    f8=open(results_path,'a')
    s8=str(acc8)
    s8=s8.replace(']','')
    s8=s8.replace('[','')  
    f8.write(s8)
    f8.close()

