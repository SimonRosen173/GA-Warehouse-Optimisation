from GA import multi_objective
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parse_args = "pop_size,n_generations,mut_tile_size,mut_tile_no," \
                 "pop_init_mode,pop_init_p," \
                 "n_agents,n_timesteps,n_cores," \
                 "cluster_node,run_notes,run_name," \
                 "wandb_mode,log_interval,save_interval," \
                 "log_folder_path,log_name"
    parse_args = parse_args.split(",")

    for parse_arg in parse_args:
        parser.add_argument(parse_arg)
    args = parser.parse_args()

    pop_size = int(args.pop_size)
    n_generations = int(args.n_generations)

    n_agents = int(args.n_agents)
    n_timesteps = int(args.n_timesteps)

    n_cores = int(args.n_cores)

    mut_tile_size = int(args.mut_tile_size)
    mut_tile_no = int(args.mut_tile_no)

    pop_init_mode = args.pop_init_mode
    pop_init_p = float(args.pop_init_p)

    run_notes = args.run_notes
    run_name = args.run_name
    cluster_node = args.cluster_node

    wandb_mode = args.wandb_mode
    log_interval = int(args.log_interval)
    save_interval = int(args.save_interval)

    log_name = args.log_name
    log_folder_path = args.log_folder_path

    if wandb_mode != "disabled":
        using_wandb = True
    else:
        using_wandb = False

    print(f"##############################")
    print(f"#        ARGUMENTS           #")
    print(f"##############################")
    print(f"pop_size={pop_size}, n_generations={n_generations}, n_agents={n_agents}")
    print(f"n_timesteps={n_timesteps}, mut_tile_size={mut_tile_size}, mut_tile_no={mut_tile_no}, "
          f"n_cores={n_cores}")
    print(f"using_wandb={using_wandb}, wandb_mode={wandb_mode}, log_interval={log_interval}, "
          f"save_interval={save_interval}")
    print(f"log_folder_path={log_folder_path}, log_name={log_name}")
    print(f"cluster_node={cluster_node}")
    print(f"run_notes={run_notes}, run_name={run_name}, tags=[\"run_1\"]")
    print(f"pop_init_mode={pop_init_mode}, pop_init_p={pop_init_p}")
    print(f"##############################\n")

    multi_objective.train(pop_size, n_generations, n_agents,
                          n_timesteps, mut_tile_size, mut_tile_no, n_cores,
                          using_wandb, wandb_mode, log_interval, save_interval,
                          log_folder_path, log_name,
                          cluster_node,
                          run_notes, run_name, ["run_1"],
                          pop_init_mode=pop_init_mode, pop_init_p=pop_init_p)
