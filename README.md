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
1. Run from Command Line Prompt: LFL-KIN.py (Q) (KinFaceW-I|KinFaceW-II) (fs|ms|fd|md)
 OR
2. Run using the batch file: RunCode.bat, this will run all the tests automatically.

# Important
Before running the code make sure to:
1. Download the KinFaceW data-sets from these links: http://www.kinfacew.com/dataset/KinFaceW-I.zip
http://www.kinfacew.com/dataset/KinFaceW-II.zip
2. Unzip the two files somewhere in your hard disk
3. Go to LoadData.py and set RootDir to the path of the directory containing the two data sets: KinFaceW-I and KinFaceW-II
