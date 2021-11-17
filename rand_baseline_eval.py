from GA import baseline_random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parse_args = "pop_size,n_generations," \
                 "n_agents,n_timesteps,n_cores," \
                 "cluster_node,run_notes,run_name," \
                 "wandb_mode,log_interval,save_interval"
    parse_args = parse_args.split(",")

    for parse_arg in parse_args:
        parser.add_argument(parse_arg)
    args = parser.parse_args()

    pop_size = int(args.pop_size)
    n_generations = int(args.n_generations)

    n_agents = int(args.n_agents)
    n_timesteps = int(args.n_timesteps)

    n_cores = int(args.n_cores)

    run_notes = args.run_notes
    run_name = args.run_name
    cluster_node = args.cluster_node

    wandb_mode = args.wandb_mode
    log_interval = int(args.log_interval)
    save_interval = int(args.save_interval)

    if wandb_mode != "disabled":
        using_wandb = True
    else:
        using_wandb = False

    baseline_random.train(pop_size, n_generations, n_agents,
                 n_timesteps, n_cores,
                 using_wandb, wandb_mode, log_interval, save_interval,
                 cluster_node,
                 run_notes, run_name, ["run_1"])
