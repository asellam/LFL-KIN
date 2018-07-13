FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-I fs
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-I fs
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-I fs
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-I fs

FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-I fd
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-I fd
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-I fd
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-I fd

FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-I ms
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-I ms
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-I ms
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-I ms

FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-I md
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-I md
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-I md
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-I md


FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-II fs
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-II fs
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-II fs
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-II fs

FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-II fd
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-II fd
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-II fd
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-II fd

FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-II ms
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-II ms
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-II ms
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-II ms

FOR /L %%F IN (2,2,32) DO LFL-KIN.py %%F KinFaceW-II md
FOR /L %%F IN (40,8,128) DO LFL-KIN.py %%F KinFaceW-II md
FOR /L %%F IN (256,128,1024) DO LFL-KIN.py %%F KinFaceW-II md
FOR /L %%F IN (1536,512,4096) DO LFL-KIN.py %%F KinFaceW-II md
PAUSE
