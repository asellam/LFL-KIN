#---------------------------
#- Kinship dataset loader  -
#---------------------------
#- By: Abdellah SELLAM     -
#-     Hamid AZZOUNE       -
#---------------------------
#- Created: 2018-04-08     -
#- Last update: 2018-07-06 -
#---------------------------

import numpy as np
import scipy.io as sio
from scipy import misc
#Change the value of this Variable to the path of the directory containing KinFaceW-I
# and KinFaceW-II folders
RootDir="../../"
#This is a dictionnary that converts kinship prefix to the sub-directory containing
# images of that particular kinship relation
PrefixToDir={"fd":"father-dau","fs":"father-son","md":"mother-dau","ms":"mother-son"}

#This function loads one fold of Kinship Pairs
#Parameters:
#-----------
# KinSet: The kinship dataset name (KinFaceW-I, KinFaceW-II)
# KinShip: The prefix of the kinship (fd: father-daughter,
#                                    fs: father-son,
#                                    md: mother-daughter,
#                                    ms: mother-son)
# Fold: Number of fold to load from five folds (1,2,3,4,5)
# Mode: Mode of images to load (0: grayscale, 1: rgb)
def LoadFold(KinSet,KinShip,Fold,Mode=1):
    #Loads a mat file (matlab variable file): the variable containing pairs used
    # for five-fold cross validation
    meta=sio.loadmat(RootDir+"/"+KinSet+"/meta_data/"+KinShip+"_pairs.mat")
    #Get The pairs part of this variable
    pairs=meta['pairs']
    #Output Variables
    TrainX=[]
    TrainY=[]
    TestX=[]
    TestY=[]
    #Path to the directory containing images
    pDir=RootDir+"/"+KinSet+"/images/"+PrefixToDir[KinShip]+"/"
    #Go through all pairs
    for p in pairs:
        #Check images' loading-mode (rgb or grayscale)
        if Mode==1:
            #Read parent's facial image
            pImg=misc.imread(pDir+p[2][0])
            #Read child's facial image
            cImg=misc.imread(pDir+p[3][0])
        else:
            #Read parent's facial image
            pImg=misc.imread(pDir+p[2][0],flatten=True)
            #Read child's facial image
            cImg=misc.imread(pDir+p[3][0],flatten=True)
        #Checkif this fold is a Train or a Test fold
        if p[0][0][0]==Fold:
            #Add fold's data to the test data
            TestX.append([pImg,cImg])
            TestY.append(p[1][0][0])
        else:
            #Add fold's data to the train data
            TrainX.append([pImg,cImg])
            TrainY.append(p[1][0][0])
    #Return Train features,Train labels, Test features and Test lables
    return (np.array(TrainX),np.array(TrainY),np.array(TestX),np.array(TestY))

#This function loads one fold of Kinship Pairs of grayscale images
#This is the version of dataset loading algorithm used in this paper
#Parameters:
#-----------
# KinSet: The kinship dataset name (KinFaceW-I, KinFaceW-II)
# KinShip: The prefix of the kinship (fd: father-daughter,
#                                    fs: father-son,
#                                    md: mother-daughter,
#                                    ms: mother-son)
# Fold: Number of fold to load from five folds (1,2,3,4,5)
def LoadFoldGrayScale(KinSet,KinShip,Fold):
    #Load grayscale pairs
    (TrainX,TrainY,TestX,TestY)=LoadFold(KinSet,KinShip,Fold,0)
    #Normalizing Train Inputs
    P0=[]#Parent image
    C0=[]#Child images
    s=TrainX.shape#Size of the list in different dimensions
    for i in range(s[0]):
        #Normalizing parent's image pixel-values between 0.0 and 1.0
        Z=(TrainX[i][0].flatten())/255.0
        #Normalizing child's image pixel-values between 0.0 and 1.0
        W=(TrainX[i][1].flatten())/255.0
        P0.append([Z])
        C0.append([W])
    #Normalizing Test Inputs
    P1=[]#Parent images
    C1=[]#Child images
    s=TestX.shape#Size of the list in different dimensions
    for i in range(s[0]):
        #Normalizing parent's image pixel-values between 0.0 and 1.0
        Z=(TestX[i][0].flatten())/255.0
        #Normalizing child's image pixel-values between 0.0 and 1.0
        W=(TestX[i][1].flatten())/255.0
        P1.append([Z])
        C1.append([W])
    #Returns: Train parent images, train child images, train labels,
    #         Test parent images, test child images, test labels
    return (np.array(P0),np.array(C0),np.array(TrainY),np.array(P1),np.array(C1),np.array(TestY))

def SaveToCSV(M,FileName):
    s=M.shape
    file=open(FileName,"w")
    for i in range(s[0]):
        Line="%f"%(M[i][0])
        for j in range(1,s[1]):
            Line=Line+";%f"%(M[i][j])
        file.write(Line+"\n")
    file.close()
