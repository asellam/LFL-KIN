FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-I fs
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-I fs
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-I fs
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-I fs

FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-I fd
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-I fd
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-I fd
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-I fd

FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-I ms
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-I ms
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-I ms
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-I ms

FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-I md
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-I md
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-I md
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-I md


FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-II fs
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-II fs
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-II fs
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-II fs

FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-II fd
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-II fd
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-II fd
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-II fd

FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-II ms
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-II ms
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-II ms
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-II ms

FOR /L %%Q IN (2,2,32) DO LFL-KIN.py %%Q KinFaceW-II md
FOR /L %%Q IN (40,8,128) DO LFL-KIN.py %%Q KinFaceW-II md
FOR /L %%Q IN (256,128,1024) DO LFL-KIN.py %%Q KinFaceW-II md
FOR /L %%Q IN (1536,512,4096) DO LFL-KIN.py %%Q KinFaceW-II md
PAUSE
