import os
import argparse
from utils.result_tools import plot_results, print_summary


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=["Rooms8", "Rooms", "FourRooms", "EightRooms", "all"], default="all")
    parser.add_argument('--results', type=str, help="path to results folder", default="results")
    parser.add_argument('--outdir', type=str, help="output directory for plots", default="results")
    parser.add_argument('--format', type=str, choices=["pdf", "png"], default="png")
    parser.add_argument('--print', action="store_true", help="only print result summary to console")

    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    if args.env == "all":
        print_summary(results_dir=args.results)
        if not args.print:
            for env in ["FourRooms", "EightRooms"]:
                plot_results(results_dir=args.results, env_name=env,
                             outfile=os.path.join(args.outdir, f"plot_{env.lower()}.png"))
    else:
        print_summary(results_dir=args.results, env_name=args.env)
        if not args.print:
            plot_results(results_dir=args.results, env_name=args.env,
                         outfile=os.path.join(args.outdir, f"plot_{args.env.lower()}.{args.format.lower()}"))
