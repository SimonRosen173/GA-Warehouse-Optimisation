from GA import multi_objective
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parse_args = "pop_size,n_generations,mut_tile_size,mut_tile_no," \
                 "cluster_node,run_notes,run_name," \
                 "log_interval,save_interval"
    parse_args = parse_args.split(",")

    for parse_arg in parse_args:
        parser.add_argument(parse_arg)
    args = parser.parse_args()

    pop_size = int(args.pop_size)
    n_generations = int(args.n_generations)

    # n_agents = int(args.n_agents)
    # n_timesteps = int(args.n_timesteps)
    n_agents = 60
    n_timesteps = 600

    mut_tile_size = int(args.mut_tile_size)
    mut_tile_no = int(args.mut_tile_no)

    run_notes = args.run_notes
    run_name = args.run_name
    cluster_node = args.cluster_node

    log_interval = int(args.log_interval)
    save_interval = int(args.save_interval)
    using_wandb = True

    multi_objective.train(pop_size, n_generations, n_agents,
                 n_timesteps, mut_tile_size, mut_tile_no,
                 using_wandb, log_interval, save_interval,
                 cluster_node,
                 run_notes, run_name, ["run_1"])
