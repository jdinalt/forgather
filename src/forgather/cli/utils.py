import os


def add_output_arg(parser):
    """Add output file argument to parser"""
    parser.add_argument(
        "-o",
        "--output-file",
        type=os.path.expanduser,
        default=None,
        help="Write output to file",
    )


def write_output(args, data):
    """For commands with an '-o' argument, either write data to stdout or to file"""
    if args.output_file:
        try:
            with open(args.output_file, "w") as f:
                f.write(data)
            print(f"Wrote output to {args.output_file}")
        except Exception as e:
            print(e)
    else:
        print(data)
