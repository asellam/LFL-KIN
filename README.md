# LFL-KIN
Linear Feature Learning for Kinship Verification
Implementation of the code for the paper:
Abdellah SELLAM and Hamid AZZOUNE, Linear Feature Learning for Kinship verification in the wild

# Prerequisites
1. Python
2. Tensorflow (pip install tensorflow-gpu)
3. Numpy (pip install numpy)
4. Scipy (pip install scipy)

# How To Use
## Finding the number of feature
The sub-directory **Find-Number-Of-Features** is dedicated for this
This script is designed to run from a shell batch file to avoid memory garbage-collecting errors (memory allocated by tensorflow not getting cleaned)
1. Run from Command Line Prompt: LFL-KIN.py (Q) (KinFaceW-I|KinFaceW-II) (fs|ms|fd|md)
 OR
1. Run using the batch file: RunCode.bat, this will run all the tests automatically. RunCode.bat is an example of a batch file (Tested on windows)
## Test the feature extraction matrix using SVM
The sub-directory **Test-Using-SVM** is dedicated for this
1. Use **Find-Number-Of-Features/LFL-KIN.py** to find the best number of Features Q'
1. Use **Find-Number-Of-Features/LFL-KIN.py** again with Q=Q' to obtain the feature extraction matrices for different folds
1. Put the obtained feature extraction matrices in this directory **Test-Using-SVM**
1. Modify the Variables: Kinset (line 19) and Kinship (line 21)
1. Run the code
1. The program will return the performance of SVM's NuSVC with different nu values.
## Test the feature extraction matrix using SVM

# Important
Before running the code make sure to:
1. Download the KinFaceW data-sets from these links: http://www.kinfacew.com/dataset/KinFaceW-I.zip
http://www.kinfacew.com/dataset/KinFaceW-II.zip
2. Unzip the two files somewhere in your hard disk
3. Go to LoadData.py and set RootDir to the path of the directory containing the two data sets: KinFaceW-I and KinFaceW-II
