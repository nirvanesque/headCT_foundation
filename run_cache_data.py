import subprocess

batch_size = 1000
length = 22158

for start_idx in range(0, length + 1, batch_size):
    end_idx = min(start_idx + batch_size, length + 1)

    cmd = '''#!/bin/bash

#SBATCH --partition=cpu_dev,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=./logs/log_{}-{}.out

source activate head_ct

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
'''.format(start_idx, end_idx)

    cmd += 'python cpu_caching.py --start_idx {} --end_idx {}'.format(start_idx, end_idx)
    with open("./jobs_{}_{}.sh".format(start_idx, end_idx), 'w') as f:
        f.write(cmd)
    result = subprocess.run("sbatch ./jobs_{}_{}.sh".format(start_idx, end_idx), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)