import argparse
import os
import subprocess

TOTAL_CPUS_PER_MACHINE = 200
CPUS_REQUIRED_PER_SEED = 80


def build_cmd(algo, config_name, seed):
    scenario_name = config_name
    if scenario_name in ['3v1', 'ca_easy', 'corner']:
        image_name = "fw/marl-gpu-fb"
        env_name = "fb"
    elif scenario_name in ['2c64zg', '2m1z']:
        image_name = "fw/marl-gpu-smac"
        env_name = "smac"
    else:
        raise NotImplementedError()

    cmd_list = [
        'python3',
        'main.py',
        f'--config {scenario_name}',
        f"--algo {algo}",
        f'--n_iterations {10 if env_name == "fb" else 4}',
        '--use_wandb',
        f'--wandb_project {algo}-{env_name}',
        f"--seed {seed}",
    ]
    if algo == 'sipo-wd':
        cmd_list = list(filter(lambda x: ("n_iterations" not in x), cmd_list))
        if scenario_name == '3v1':
            eps = 2e-3
            ir_scaling = 0.1
            ll_max = 10
        elif scenario_name == '2m1z' or scenario_name == '2c64zg':
            eps = 3e-3
            ir_scaling = 5.0 if scenario_name == '2m1z' else 2.0
            ll_max = 10
        else:
            raise NotImplementedError()
        cmd_list += [
            f'--threshold_eps {eps}',
            f"--ll_max {ll_max}",
            f'--intrinsic_reward_scaling {ir_scaling}',
            f'--wandb_group {scenario_name}eps{eps}i{ir_scaling}L{ll_max}',
            f'--n_iterations {10 if env_name == "fb" else 4}',
            # f'--archive_policy_dirs results/sipo-wd/{scenario_name}/check/{seed}/run1/iter0/model.pt',
            # f'--archive_traj_dirs results/sipo-wd/{scenario_name}/check/{seed}/run1/iter0/data.traj'
        ]
        if scenario_name == '2c64zg' or scenario_name == '3v1':
            if os.path.exists(f"results/sipo-wd/{scenario_name}/check/{seed}/run1/iter0"):
                cmd_list += [
                    f'--archive_policy_dirs results/sipo-wd/{scenario_name}/check/{seed}/run1/iter0/model.pt',
                    f'--archive_traj_dirs results/sipo-wd/{scenario_name}/check/{seed}/run1/iter0/data.traj'
                ]
    if algo == 'smerl':
        cmd_list = [
            'python3',
            'main.py',
            f'--config {scenario_name}',
            f"--algo {algo}",
            f'--n_iterations 1',
            '--use_wandb',
            f'--wandb_project {algo}-{env_name}',
            f"--seed {seed}",
        ]

    return " ".join(cmd_list)


def submit_job(algo, scenario_name, seed, num_seeds_per_machine):
    os.makedirs(f"slurm_outs/{algo}", exist_ok=True)
    if scenario_name in ['3v1', 'ca_easy', 'corner']:
        image_name = "fw/marl-gpu-fb"
        env_name = "fb"
    elif scenario_name in ['2c64zg', '2m1z']:
        image_name = "fw/marl-gpu-smac"
        env_name = "smac"
    else:
        raise NotImplementedError()

    lines = [
        "#!/bin/sh",
        f"#SBATCH --output=slurm_outs/{algo}/{scenario_name}_seed{seed}.out",
        "#SBATCH --gpus=geforce:1",
        F"#SBATCH --cpus-per-gpu={min(TOTAL_CPUS_PER_MACHINE, CPUS_REQUIRED_PER_SEED * num_seeds_per_machine)}",
        "#SBATCH --mem-per-cpu=1G",
        "#SBATCH --ntasks=1",
        f"#SBATCH --job-name={algo}-{scenario_name}-s{seed}",
        "#SBATCH --partition=cpu",
    ]

    srun_flags = [
        f"--ntasks={num_seeds_per_machine}",
        f"--cpus-per-task={min(TOTAL_CPUS_PER_MACHINE, CPUS_REQUIRED_PER_SEED * num_seeds_per_machine) // num_seeds_per_machine}",
        f"--mem-per-cpu=1G",
        f"--ntasks-per-gpu={num_seeds_per_machine}",
        "--gpus=geforce:1",
        f"--container-image={image_name}",
        "--container-mounts=/home/fw/workspace/fast_mappo:/fast_mappo",
        "--container-workdir=/fast_mappo",
    ]
    if algo in ['sipo-rbf', 'sipo-wd', 'dipg', 'rspo', 'smerl']:
        cmd = build_cmd(algo, config_name=scenario_name, seed=seed)
    else:
        raise NotImplementedError()

    flags = " ".join(srun_flags)
    srun_cmd = f'srun -l {flags} {cmd}'

    lines.append(srun_cmd)

    script = '\n'.join(lines).encode('ascii')

    return subprocess.check_output(['sbatch', '--parsable'],
                                   input=script).decode('ascii').strip()


parser = argparse.ArgumentParser()
parser.add_argument('--algo', '-g', type=str)
parser.add_argument('--map_names', '-e', type=str, nargs="+")
parser.add_argument('--seeds', '-s', type=int, nargs="+")
parser.add_argument("--num_seeds_per_machine", type=int, default=1)


def main():
    args = parser.parse_args()
    assert args.algo in ['sipo-rbf', 'sipo-wd', 'dipg', 'rspo', 'smerl']
    jobs = []
    for map_name in args.map_names:
        assert map_name in ['3v1', 'ca_easy', 'corner', '2m1z', '2c64zg']
        num_seeds_per_machine = args.num_seeds_per_machine
        for seed in args.seeds:
            r = submit_job(args.algo, map_name, seed, num_seeds_per_machine)
            jobs.append(r)
    print(
        f"Algorithm: {args.algo}. Map names {args.map_names}. Seeds: {args.seeds}. "
        f"Submitted slurm job IDs: {','.join(jobs)}.")


if __name__ == '__main__':
    main()
    # print(build_cmd('sipo-wd', '3v1', 4))
