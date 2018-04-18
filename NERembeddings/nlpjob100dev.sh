#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=nlp_job100
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --mail-type=END
#SBATCH --mail-user=lh1036@nyu.edu
 
module purge
module load python3/intel/3.6.3
RUNDIR=$SCRATCH/my_project/run-${SLURM_JOB_ID/.*} 
mkdir -p $RUNDIR

echo $RUNDIR

cp nametagger_embeddings_with_geotext.py $RUNDIR
cp CONLL_train.pos-chunk-name $RUNDIR
cp CONLL_dev.pos-chunk $RUNDIR
cp glove50d.txt $RUNDIR
cp -r py3.5/ $RUNDIR

export RUNDIR

cd $RUNDIR

ls

source py3.5/bin/activate

python3 nametagger_embeddings_with_geotext.py CONLL_train.pos-chunk-name CONLL_dev.pos-chunk glove50d.txt 100 -o response100i.name

exit
