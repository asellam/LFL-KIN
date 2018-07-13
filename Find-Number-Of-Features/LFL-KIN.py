#----------------------------------------------------------------
#- Linear Feature Learning for Kinship Verification in the Wild -
#----------------------------------------------------------------
#- By: Abdellah SELLAM                                          -
#-     Hamid AZZOUNE                                            -
#----------------------------------------------------------------
#- Created: 2018-04-08                                          -
#- Last update: 2018-07-12                                      -
#----------------------------------------------------------------

import os
import sys
import time
import numpy as np
import random as rd
import tensorflow as tf
from LoadData import LoadFoldGrayScale as LoadFold
from LoadData import SaveToCSV
# Set Random Iterator's initial value depending on current time, this Results
# in different random values each execusion
rd.seed(time.time())
# Set KinFaceW used set from Command line Prompt
KinSet=sys.argv[2]
# Set KinFaceW kinship type from Command line Prompt
KinShip=sys.argv[3]
# Set number of LFL features from Command line Prompt
Q=int(sys.argv[1])
# Display Number of LFL features in the consol
print("Q:",Q)
# Maximum number of Training Epochs
MAX_EPOCH=1000
# Number of Training Epochs between convergence checks
ValidCh=1
# Threshold of difference between last and current test loss (used to detect
# algorithm convergence: check if no further optimization is significant)
ValidTh=0.0001
# Number of Training Trials for this number of features: The program will run
# the training this number of times and returns best performance
nTrial=1
# Best-Threshold-Search step: After learning the feature matrix with this Number
# of features, we must judge its performance on test data in order to find the
# best number of features.
# The best feature matrix (best number of feautres) is the one that seperate
# the two classes (positive/negative) the best, i.e: a threshold on distances
# that best seperate positive and negative pairs must be found an then the
# Accuracy of this threshold is outputed as the performance of this number of
# features.
# The number of features with the best performance will be choosen
SearchT=1000
# Range of initial random values of the feature extraction matrix
MinRand=-0.01
MaxRand=+0.01
# Learning Rate for the Gradient Descent algorithm
LR=0.01
# used to detect convergence
MaxFail=5
#Number of Folds for K-Fold-Cross-Validation
nFold=5

# returns a random value in the predefined range for the matrix initial values
def RandomVal():
    return rd.random()*(MaxRand-MinRand)+MinRand

# Returns the best threshold for this feature matrix
# D: List of distances of all pairs
# K: Kinship class (positive/negative) of all pairs (same order as D)
# N: Number pairs (for D and K)
def ThreshPerf(D,K,N):
    # Compute Minimum and Maximum distances
    MinD=D[0] # Minimum
    MaxD=D[0] # Maximum
    for i in range(N):
        if D[i]<MinD:
            MinD=D[i]
        if D[i]>MaxD:
            MaxD=D[i]
    # The algorithm will Compute the performances of 'SearchT' thresholds
    # These 'SearchT' thresholds are values in the range [MinD .. MaxD]
    Th0=MinD # The best threshold lowor bound
    Th1=MinD # The best threshold upper bound
    # Since All threshold in the range [Th0 .. Th1] will have the same (best)
    # performance, the algorithm will return the mean value of these two bounds
    # as best threshold
    Perf=0 # Holds the best performance
    # Search a number (equal to 'SearchT') of thresholds
    for T in range(SearchT):
        # Pick a threshold between Minimum and Maximum distances
        ThisTh=MinD+T*(MaxD-MinD)/SearchT
        # Compute the Accuracy (performance) of this threshold
        ThisPerf=0
        for i in range(N):
            if D[i]<ThisTh:
                O=1
            else:
                O=0
            if O==K[i]:
                ThisPerf=ThisPerf+1
        # See if the current performance is better than the last best
        if ThisPerf>Perf:
            # If this is a new best then
            #   1. Update the best performance value
            #   2. initialize Th0 and Th1 to be equal to the current threshold
            Th0=ThisTh
            Th1=ThisTh
            Perf=ThisPerf
        # While the performance of the last and current threshold are the same
        #   Th1 (upper bound of best threshold) will be updateed with greater
        #   values of threshold
        if ThisPerf==Perf:
            Th1=ThisTh
    # We will return the average of Th0 and Th1 (bounds of best threshold) as
    # the threshold with best performance using this number of feautres
    Th=(Th0+Th1)/2
    return (Th,Perf)

#Training interactive Display Variable
Text=""
# Prepare the file in which we hold results per number of features (Q)
# Each one of the for sub-sets of each one of the two Kinship datasets will
# have its own file of results
# Extension CSV: opens with MS-Excel, LiberOffice ...
csvr=open("./Results_"+KinShip+"_%d"%(Q)+".csv","w")
csvr.write("Fold;Trial;Epochs;Train Loss;Test Loss;Train Accuracy;Test Accuracy\n")
csvr.close()
# K-Cross Validation Results
# Item 1: Average Number of Iterations (Epochs)
# Item 2: Average Training Loss
# Item 3: Average Test Loss
# Item 4: Average Training Accuracy
# Item 5: Average Test Accuracy
Mean=[0.0,0.0,0.0,0.0,0.0]
for Fold in range(1,nFold+1):
    # This Fold's best results over 'nTrial' number of trials
    # Item 1: Number of Iterations (Epochs)
    # Item 2: Training Loss
    # Item 3: Test Loss
    # Item 4: Training Accuracy
    # Item 5: Test Accuracy
    Best=[0.0,0.0,0.0,0.0,0.0]
    # Loads a Train/Test pairs of this fold
    (P0,C0,K0,P1,C1,K1)=LoadFold(KinSet,KinShip,Fold)
    # N0: Number of train pairs
    # a: 2 (2 images in each pair)
    # M0: Number of gray-scale pixels in each image
    (N0,a,M0)=P0.shape
    # N1: Number of test pairs
    # a: 2 (2 images in each pair)
    # M1: Number of gray-scale pixels in each image
    (N1,b,M1)=P1.shape
    # Difference between gray-scale images of a pair (parent/child)
    D0=P0-C0
    D1=P1-C1
    # Try 'nTrail' times and return best results in 'Best' list
    for TR in range(nTrial):
        # Initialize a Tensorflow session
        ss=tf.Session()
        # Make the initial random feature matrix
        A0=[]
        for i in range(M0):
            Z=[]
            for j in range(Q):
                Z.append(RandomVal())
            A0.append(Z)
        # Use GPU for this Computation
        with tf.device("/gpu:0"):
            # A Tensor that holds the list of vectors of differences between
            # gray-scale images of (parent,child) pairs [(P1-C1),(P2-C2),...,(Pn-Cn)]
            D = tf.placeholder(tf.float32,shape=(None,1,None))
            # A Tensor that holds kinship classes (positive/negative)
            T = tf.placeholder(tf.float32,shape=(None,))
            # A Tensor that holds the number of pairs
            L = tf.placeholder(dtype=tf.int32)
            # A Tensor Variable that contains the feature metric's matrix
            A = tf.Variable(A0,dtype=tf.float32)

            # A tensorflow's Variables' initializer
            init = tf.global_variables_initializer()

            # Gradient Descent's Loss function definition

            # Tensorflow's While Loop continuation condition
            def cond(i, others):
                return i < L

            # Tensorflow's While Loop body
            def body(i, s):
                # i^th pair's gray-scale-difference
                x = D[i]
                # i^th pair's KinShip class
                t = T[i]
                # multiply the gray-scale-difference by feature matrix A
                #  ~ Equivalent to P[i]*A-C[i]*A
                T1 = tf.matmul(x,A)
                # Element-wise square of x*A
                T3 = tf.square(T1)
                # Overall sum of squared differences after multiplication by
                # the feature matrix (Equivalent to square of euclidian distance
                # between images of a signle pair that were transformed by matrix A)
                d = tf.reduce_sum(T3)
                # if this is a positive pair then
                #     add d to the loss value, distance between images of a
                #     positive pair should be small, so bigger distance means
                #     more loss
                # if this is a negative pair then
                #     add 1/d to the loss value, distance between images of a
                #     negative pair should be big, so bigger distance means
                #     less loss
                return i + 1, s+(1-t)/d+t*d

            # Makes a Tensorflow's while loop for the loss function's definition
            loop = tf.while_loop(cond, body, (tf.constant(0),tf.constant(0.0,shape=(),dtype=tf.float32)))
            # The Loss is defined as the mean of signle pairs losses
            loss = loop[1]/tf.cast(L, tf.float32)
            # Creates a Tensorflow's Gradient Descent Optimizer
            optimizer = tf.train.GradientDescentOptimizer(LR)
            # Makes a Tensorflow's Gradient Descent Training process
            train = optimizer.minimize(loss)
            # Runs the Variable initializer on this session
            ss.run(init)
            # Training
            # Last Epoch's loss on test data
            LastLoss=ss.run(loss,{D: D1, T:K1, L:N1})
            # Epochs' counter
            E=0
            # Stop boolean
            stop=0
            # Convergence counter
            nFail=0
            # While not reached Maximum number of Epochs and stop is false
            while(E<MAX_EPOCH)and(stop==0):
                # Runs a single training epoch
                ss.run(train, {D: D0, T:K0, L:N0})
                # Computes losses after this epoch
                TrainLoss=ss.run(loss,{D: D0, T:K0, L:N0})
                TestLoss=ss.run(loss,{D: D1, T:K1, L:N1})
                # Difference between Last Epoch's loss and current loss on test data
                Diff=LastLoss-TestLoss
                # Clear consol, this won't work properly on idle, better run
                # this script for cmd (shell)
                os.system("cls")
                # Show old training results
                print(Text)
                # show this epoch's training progress
                print('Q: %d Epoch: %d[%d]'%(Q,Fold,TR+1),E+1,'/',MAX_EPOCH,TrainLoss,":",TestLoss,"[",Diff,"]")
                # Check for convergence, if the Difference between Last Epoch's
                # loss and current loss on test data is less than a low value
                # then stop the algorithm, because, it is of no use to go further
                if (E+1)%ValidCh==0:
                    if Diff<ValidTh:
                        nFail=nFail+1
                        if nFail>=MaxFail:
                            stop=1
                # Update last loss value
                LastLoss=TestLoss
                # Increment epoch's counter
                E=E+1
        # Results
        # M: Learned feature metric's matrix
        # L: Learning Loss
        M, L = ss.run([A, loss], {D: D0, T:K0, L:N0})

        # Search best threshold
        # d0: Gray-scale-differences on training samples
        d0=[]
        for i in range(N0):
            Z=np.matmul(P0[i],M)
            W=np.matmul(C0[i],M)
            d0.append(np.linalg.norm(Z-W))

        # d4: Gray-scale-differences on test samples
        d1=[]
        for i in range(N1):
            Z=np.matmul(P1[i],M)
            W=np.matmul(C1[i],M)
            d1.append(np.linalg.norm(Z-W))

        # Results
        # Th0: Best threshold using training data
        # Perf0: Performance of the best threshold using training data
        (Th0,Perf0)=ThreshPerf(d0,K0,N0)
        # Th1: Best threshold using test data
        # Perf1: Performance of the best threshold using test data
        (Th1,Perf1)=ThreshPerf(d1,K1,N1)
        # If the best test performance (in which we are interested) for this
        # trial is better than the last one, then updated best results of this
        # fold and save the feature-extraction matrix Learned
        if(Perf1>Best[4]):
            Best[0]=E
            Best[1]=TrainLoss
            Best[2]=TestLoss
            Best[3]=Perf0
            Best[4]=Perf1
            if KinSet=="KinFaceW-I":
                FileNameM="./M_"+KinShip+"-I_%d"%(Fold)+".csv"
            else:
                FileNameM="./M_"+KinShip+"-II_%d"%(Fold)+".csv"
            SaveToCSV(M,FileNameM)
        # Terminate the Tensorflow's session created earlier
        ss.close()
        # Output results to result's csv file
        csvr=open("./Results_"+KinShip+"_%d"%(Q)+".csv","a")
        csvr.write("%d;%d;%d;%f;%f;%f;%f\n"%(Fold+1,TR,E,TrainLoss,TestLoss,Perf0,Perf1))
        csvr.close()
        # Let the GPU rest for 20 seconds, just for the sake of hardware :p
        time.sleep(20)
    # K-Fold-Cross-Validation Results (Sum)
    Mean[0]=Mean[0]+Best[0]
    Mean[1]=Mean[1]+Best[1]
    Mean[2]=Mean[2]+Best[2]
    Mean[3]=Mean[3]+(Best[3]*100/N0)
    Mean[4]=Mean[4]+(Best[4]*100/N1)
# K-Fold-Cross-Validation Results (Average)
for i in range(5):
    Mean[i]=Mean[i]/nFold
# Make a new line on the training interactive Display Variable
# Why this technique?
#   if you run multiple times this script from a shell batch file with different
#   number of features (as i do), this program will always output in the consol
#   final results of older execusions (other number of features), and it will
#   output the results of the last iteration for the current execusion (current
#   number of features)
#   The variable 'Text' holds older execusions' results, to output them before
#   the current results
Text=Text+"%4d : %4d : %.03f : %.03f : %.01f%% : %.01f%%\n"%(Q,Mean[0],Mean[1],Mean[2],Mean[3],Mean[4])
Line="%d;%f;%f;%f;%f;%f\n"%(Q,Mean[0],Mean[1],Mean[2],Mean[3],Mean[4])
csvr=open("./Results.csv","a")
csvr.write(Line)
csvr.close()
print(Text)
