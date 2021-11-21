# Code to submit multiple pbs jobs via qsub

import os

tmp_dir = "tmp/qsub_files/"


def clear_tmp_qsub_dir():
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))


def get_template_str():
    with open("template.qsub", "r") as f:
        file_str = f.read()

    return file_str


def create_qsub(template_str, qsub_file_name, resource_reqs, partition,
                pbs_log_name, log_name, log_folder_path, job_name,
                script, script_args):

    qsub_str = template_str
    args_str = " ".join(script_args)

    replace_str_dict = {
        "wall_time": resource_reqs["wall_time"],
        "n_cpus": resource_reqs["n_cpus"],
        "mem": resource_reqs["mem"],
        "partition": partition,
        "pbs_log_name": pbs_log_name,
        "job_name": job_name,
        "log_name": log_name,
        "log_folder_path": log_folder_path,
        "script": script,
        "args": args_str
    }

    for key in replace_str_dict.keys():
        val = replace_str_dict[key]
        key_str = "{"+key+"}"
        qsub_str = qsub_str.replace(key_str, str(val))

    with open(tmp_dir + qsub_file_name, "w") as f:
        f.write(qsub_str)

    return tmp_dir + qsub_file_name


def execute_qsub_file(file_path):
    os.system(f"qsub \"{file_path}\"")


def submit_multiple_jobs(n_jobs, resource_reqs, partition,
                         pbs_log_name_prefix, log_name_prefix, log_folder_path,
                         job_name_prefix,
                         script, script_args):
    assert n_jobs <= 10, "n_jobs cannot be greater than 10."

    orig_script_args = script_args.copy()
    clear_tmp_qsub_dir()
    template_str = get_template_str()

    # if {job_id} is in args then replace that with current job_id
    for job_id in range(n_jobs):
        script_args = [arg.replace("{job_id}", str(job_id)) for arg in orig_script_args]
        qsub_file_name = f"job_{job_id}.qsub"
        pbs_log_name = f"{pbs_log_name_prefix}_{job_id}"
        log_name = f"{log_name_prefix}_{job_id}"
        job_name = f"{job_name_prefix}_{job_id}"

        qsub_file_path = create_qsub(template_str, qsub_file_name, resource_reqs,
                                     partition, pbs_log_name, log_name, log_folder_path,
                                     job_name, script, script_args)
        execute_qsub_file(qsub_file_path)
    pass


def multi_obj_train():
    clear_tmp_qsub_dir()
    template_str = get_template_str()

    n_jobs = 2

    resource_reqs = {
        "wall_time": "72:00:00",
        "n_cpus": "24",
        "mem": "16gb",
    }
    base_name = "ga60a2"

    pop_size = 64
    n_generations = 4000
    mut_tile_size = 2
    mut_tile_no = 1
    n_agents = 60
    n_timesteps = 500
    n_cpus = resource_reqs["n_cpus"]
    cluster_node = -1
    run_notes = """Notes"""
    run_name = "\"" + base_name + ".{job_id}\""
    wandb_mode = "\"offline\""
    log_interval = 1
    save_interval = 50
    log_folder_path = "/mnt/lustre/users/srosen/logs"
    log_name = "\"" + base_name + "_{job_id}\""

    args = [pop_size, n_generations, mut_tile_size, mut_tile_no, n_agents, n_timesteps, n_cpus, cluster_node,
            run_notes, run_name, wandb_mode, log_interval, save_interval, "\"" + log_folder_path + "\"",
            log_name]
    args = [str(arg) for arg in args]

    # 16 20 2 1 5 600 4 -1 "Test" "Test" "disabled" 1 1 "Logs" "test"
    # args = f"64 1000 2 1 60 600 {resource_reqs['n_cpus']} -1 ""Run"" ""Run {job_id}"" ""offline"" 1 50".split(" ")

    submit_multiple_jobs(n_jobs, resource_reqs, partition="smp",
                         pbs_log_name_prefix=base_name, log_name_prefix=base_name,
                         log_folder_path=log_folder_path,
                         job_name_prefix=base_name,
                         script="multi_obj_train.py", script_args=args)


def rand_evals():
    clear_tmp_qsub_dir()
    template_str = get_template_str()

    n_jobs = 10

    resource_reqs = {
        "wall_time": "24:00:00",
        "n_cpus": "24",
        "mem": "16gb",
    }
    base_name = "rand60a2"

    pop_size = 64
    n_generations = 400
    n_agents = 60
    n_timesteps = 500
    n_cpus = resource_reqs["n_cpus"]
    cluster_node = -1
    run_notes = """Notes"""
    run_name = "\"" + base_name + ".{job_id}\""
    wandb_mode = "\"offline\""
    log_interval = 1
    save_interval = 50
    log_folder_path = "/mnt/lustre/users/srosen/rand_logs"
    log_name = "\"" + base_name + "_{job_id}\""

    args = [pop_size, n_generations, n_agents, n_timesteps, n_cpus, cluster_node,
            run_notes, run_name, wandb_mode, log_interval, save_interval, "\"" + log_folder_path + "\"",
            log_name]
    args = [str(arg) for arg in args]

    # 16 20 2 1 5 600 4 -1 "Test" "Test" "disabled" 1 1 "Logs" "test"
    # args = f"64 1000 2 1 60 600 {resource_reqs['n_cpus']} -1 ""Run"" ""Run {job_id}"" ""offline"" 1 50".split(" ")

    submit_multiple_jobs(n_jobs, resource_reqs, partition="smp",
                         pbs_log_name_prefix=base_name, log_name_prefix=base_name,
                         log_folder_path=log_folder_path,
                         job_name_prefix=base_name,
                         script="rand_baseline_eval.py", script_args=args)


if __name__ == "__main__":
    # multi_obj_train()
    rand_evals()
