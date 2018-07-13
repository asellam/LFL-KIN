import time
import numpy as np
import random as rd
from sklearn import svm
import sys
sys.path.append("D:/PhD")
import LoadData as data
KinSet="KinFaceW-II"
KinShip="md"
nFold=5
MaxNU=100


csvF=open("SVM_"+KinShip+".csv","w")
csvF.write("NU;Accuracy\n")
for NU in range(MaxNU):
    mean=0
    for Fold in range(1,nFold+1):
        if KinSet=="KinFaceW-I":
            FileNameM="M_"+KinShip+"1_%d"%(Fold)+".csv"
        else:
            FileNameM="M_"+KinShip+"2_%d"%(Fold)+".csv"
        (P0,C0,K0,P1,C1,K1)=data.LoadFoldNormal(KinSet,KinShip,Fold,(0,0,0))
        M=np.genfromtxt(FileNameM, delimiter=";")
        (N0,A,M0)=P0.shape
        (N1,B,M1)=P1.shape

        X0=[]
        Y0=[]
        for i in range(N0):
            X0.append((np.matmul(P0[i],M)-np.matmul(C0[i],M)).flatten())
            Y0.append(K0[i])
        X0=np.array(X0)
        Y0=np.array(Y0)

        X1=[]
        Y1=[]
        for i in range(N1):
            X1.append((np.matmul(P1[i],M)-np.matmul(C1[i],M)).flatten())
            Y1.append(K1[i])
        X1=np.array(X1)
        Y1=np.array(Y1)

        #print(X0.shape)
        #print(Y0.shape)
              
        model = svm.NuSVC(nu=(NU+1)/MaxNU,kernel="rbf")
        model.fit(X0, Y0)  
            
        acc=0
        W1=model.predict(X1)
        for i in range(N1):
            O=W1[i]
            K=Y1[i]
            if O==K:
                acc=acc+1
            #print(K,":",O,acc)
        print(Fold,":",acc/N1)
        mean=mean+acc/N1
    print("Overall Accuracy:",mean/nFold)
    print("NU:",(NU+1)/MaxNU,"\n")
    csvF.write("%f;%f\n"%((NU+1)/MaxNU,mean/nFold))
csvF.close()
