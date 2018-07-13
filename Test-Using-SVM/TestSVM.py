#----------------------------------------------------------------
#- Linear Feature Learning for Kinship Verification in the Wild -
#----------------------------------------------------------------
#- Features' Classification with SVM                            -
#----------------------------------------------------------------
#- By: Abdellah SELLAM                                          -
#-     Hamid AZZOUNE                                            -
#----------------------------------------------------------------
#- Created: 2018-04-08                                          -
#- Last update: 2018-07-13                                      -
#----------------------------------------------------------------

import time
import numpy as np
import random as rd
from sklearn import svm
from LoadData import LoadFoldGrayScale as LoadFold
# KinfaceW dataset (KinFaceW-I or KinFaceW-II)
KinSet="KinFaceW-II"
# KinfaceW subset (relationship: fs, fd, ms or md)
KinShip="fs"
# Number of folds in K-Fold-Cross-Validation
nFold=5
# Maximum number of SVM's Nu (for SVCnu) values to test with
MaxNU=100

# FileNameR: File to which accuracies for different values of nu are saved
# Example: if kinship subset is 'fs' and KinFaceW dataset is 'KinFaceW-II' then:
#     results will be saved to a file named: 'SVM_fs-II.csv'
# Extension CSV: opens with MS-Excel, LiberOffice ...
if KinSet=="KinFaceW-I":
    FileNameR="SVM_"+KinShip+"-I"+".csv"
else:
    FileNameR="SVM_"+KinShip+"-II"+".csv"
# Create the results' file
csvF=open(FileNameR,"w")
csvF.write("NU;Accuracy\n")
# Try a number 'MaxNU' of values for SVM parameter 'Nu'
for NU in range(MaxNU):
    # SVM's NuSVC nu value
    ThisNu=(NU+1)/MaxNU
    # K-Fold-Cross-Validation's average Accuracy
    mean=0
    # K-Fold-Cross-Validation's iterations
    for Fold in range(1,nFold+1):
        # FileNameM: the file that contains a feature extraction matrix for this
        # Fold
        # To obtain such a file:
        #     1. Run several iteration of LFL-KIN.py to find best number of
        #     features Q'
        #     2. Run LFL-KIN.py with Q=Q' for the concerned kinship subset, the
        #     program will save then a model to a csv file
        # Example: if you run with this command line:
        #     LFL-KIN.py 32 KinFaceW-I ms
        # Then: the model's file name will be:
        #     M_ms-I_1.csv for the first fold of K-Fold-Cross-Validation
        if KinSet=="KinFaceW-I":
            FileNameM="M_"+KinShip+"-I_%d"%(Fold)+".csv"
        else:
            FileNameM="M_"+KinShip+"-II_%d"%(Fold)+".csv"
        # The saved features' extraction matrix
        M=np.genfromtxt(FileNameM, delimiter=";")
        # Loads the Train/Test pairs of this fold
        # Inputs:
        #   KinSet: KinFaceW dataset (KinFaceW-I or KinFaceW-II)
        #   Kinship: KinFaceW subset (fs, fd, ms or md)
        #   Fold: K-Fold-Cross-Validation's fold
        # Outputs:
        #   P0: Gray-Scale images of parents (Training data)
        #   C0: Gray-Scale images of children (Training data)
        #   K0: Kinship label (positive/negative) (Training data)
        #   P1: Gray-Scale images of parents (Test data)
        #   C1: Gray-Scale images of children (Test data)
        #   K1: Kinship label (positive/negative) (Test data)
        (P0,C0,K0,P1,C1,K1)=LoadFold(KinSet,KinShip,Fold)
        # N0: Number of train pairs
        # A: 1
        # M0: Number of gray-scale pixels in each image
        (N0,A,M0)=P0.shape
        # N1: Number of train pairs
        # B: 1
        # M1: Number of gray-scale pixels in each image
        (N1,B,M1)=P1.shape
        # SVM Training (inputs/targets)
        X0=[]# Inputs: Distances between (parent,child) pairs of gray-scale images
        Y0=[]# Targets: KinShip labels (positive/negative)
        for i in range(N0):
            X0.append((np.matmul(P0[i],M)-np.matmul(C0[i],M)).flatten())
            Y0.append(K0[i])
        X0=np.array(X0)
        Y0=np.array(Y0)

        # SVM Test (inputs/targets)
        X1=[]# Inputs: Distances between (parent,child) pairs of gray-scale images
        Y1=[]# Targets: KinShip labels (positive/negative)
        for i in range(N1):
            X1.append((np.matmul(P1[i],M)-np.matmul(C1[i],M)).flatten())
            Y1.append(K1[i])
        X1=np.array(X1)
        Y1=np.array(Y1)

        # Creates a SVM's NuSVC classifier with RBF kernel
        model = svm.NuSVC(nu=ThisNu,kernel="rbf")
        # Fits the SVM classifier to the training data
        model.fit(X0, Y0)

        # Computes current fold's test accuracy
        acc=0 # Correctly-classified-inputs' counter
        # Predicted labels
        W1=model.predict(X1)
        # Go through all labels to Compute accuracy
        for i in range(N1):
            O=W1[i] # Predicted label
            K=Y1[i] # Ground-truth label
            if O==K:
                acc=acc+1
        # Displays the fold number and its accuracy
        print(Fold,":",acc/N1)
        # Add to the K-Fold-Cross-Validation average accuracy
        mean=mean+acc/N1
    # Overall accuracy for this fold
    print("Overall Accuracy:",mean/nFold)
    print("NU:",ThisNu,"\n")
    csvF.write("%f;%f\n"%(ThisNu,mean/nFold))
csvF.close()
