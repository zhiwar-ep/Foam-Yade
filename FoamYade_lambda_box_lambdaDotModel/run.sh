#! /bin/bash
blockMesh
cp -r 0_org 0
decomposePar
mkdir spheres
# 6. Create a symbolic link to Yade Install
#ln -s /path/to/yade/install/bin/yade-exec yadeimport.py
#ln -s /home/pawel/Programy/Yade/Yade-2412/install/bin/yade-2025-05-28.git-e608e46 yadeimport.py

#In yade serial:
#python3 scriptMPI.py
#./yadeimport.py scriptMPI.py
#/home/pawel/Programy/Yade/Yade-2412/install/bin/yade-2025-05-28.git-e608e46 scriptMPI.py

#In yade parallel
#mpirun --allow-run-as-root -n 2 /home/pawel/Programy/Yade/Yade-2412/install/bin/yade-2025-05-28.git-e608e46 scriptMPI.py

mpirun --allow-run-as-root -n 2 /home/zhiwar/yade-install/bin/yade-Unknown MPI_lambda.py | tee log.txt
#mpirun -n 2 /home/zhiwar/yade-install/bin/yade-Unknown MPI_lambda.py | tee log.txt


#mpirun --allow-run-as-root -np 2 /home/zhiwar/yade-install/bin/yade-Unknown MPI_lambda.py : -np 2 lambdaFoamYade -parallel 


#mpirun --allow-run-as-root -n 2 ./yadeimport.py scriptMPI.py
