import os
import subprocess

num_seeds = 1
# seed = 6

# for map_name in ['3v1']:
slurm_ids = []
for seed in range(1, 7):
    lines = [
        "#!/bin/sh",
        f"#SBATCH --output=ppo_seed{seed}.out",
        f"#SBATCH --container-image=fw/isaacgym",
        "#SBATCH --container-mounts=/home/fw/workspace/fast_mappo:/fast_mappo,/home/fw/sipo_archive:/sipo_archive",
        "#SBATCH --container-workdir=/fast_mappo",
        "#SBATCH --gpus=geforce:1",
        "#SBATCH --cpus-per-gpu=200",
        "#SBATCH --mem-per-cpu=1G",
        "#SBATCH --ntasks=1",
        f"#SBATCH --job-name=ppo-isaachumanoid",
        "#SBATCH --partition=cpu",
    ]

    # lines += ['ls']
    # lines += ['cd /fast_mappo/']
    # lines += ['echo "export " >> ~/.profile']
    # wandb_url = '"http://proxy.newfrl.com:8081"'
    # lines += [f"echo 'export WANDB_BASE_URL={wandb_url}' >> ~/.profile"]
    # lines += ['source ~/.profile']

    srun_flags = [
        f"--ntasks={num_seeds}",
        f"--cpus-per-task=10",
        f"--mem-per-cpu=1G",
        f"--ntasks-per-gpu={num_seeds}",
        "--gpus=geforce:1",
    ]
    cmd1 = "pip3 install -e environment/IsaacGymEnvs"
    cmd2 = " ".join([
        'python3',
        'main.py',
        f'--config humanoid',
        '--use_wandb',
        '--wandb_project isaac-humanoid-dvd',
        "--wandb_group ppo",
        "--population_size 4",
        # "--intrinsic_reward_scaling 0.5",
        f"--seed {seed}",
        # f"--wandb_name seed{seed}",
    ])
    # cmd = "ls"
    # a = "'SLURM_PROCID'"
    # srun_cmd = f'srun -l {" ".join(srun_flags)} python3 -c "import os; print(os.environ[{a}])"'
    # srun_cmd = f'srun --gpus=1 --ntasks-per-gpu=3 -c1 -n3 python3 -c "import os; print(os.environ[{a}])"'
    flags = " ".join(srun_flags)
    srun_cmd = f'srun -l {flags} bash -c "{cmd1} && {cmd2}"'

    lines.append(srun_cmd)

    script = '\n'.join(lines).encode('ascii')

    r = subprocess.check_output(['sbatch', '--parsable'],
                                input=script).decode('ascii').strip()
    slurm_ids.append(str(r))
print(",".join(slurm_ids))
