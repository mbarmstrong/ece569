###========================================
#!/bin/bash
#BSUB -n 1 
#BSUB -o lsf.out
#BSUB -e lsf.err
#BSUB -q "standard"
#BSUB -J module2 
#---------------------------------------------------------------------
#BSUB -R gpu
#BSUB -R "span[ptile=2]"
mpirun -np 1 ./DeviceQuery_Solution > output.txt
###end of script
