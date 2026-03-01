#!/bin/bash
#Set job requirements
#SBATCH -N 3 --tasks-per-node 192 -t 12:00:00 -p genoa
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=r.f.a.depunder@uva.nl
#SBATCH --ear=off


# Load modules
source ~/BLADE/BLADE_env_FINAL/bin/activate
module load 2023
module load Lumerical/2023-R2.3-OpenMPI-4.1.5


#Run program
mpirun -n 576 python 01VolatilityMain_Calc.py "$TMPDIR"/input_dir "$TMPDIR"/output_dir

