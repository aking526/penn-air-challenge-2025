import argparse
import core
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        "--p", 
        type=int, 
        choices=[1, 2, 3, 4, 5, 6], 
        help="The part of the challenge to run"
    )

    _fn = [
        core.part_1,
        core.part_2,
        core.part_3,
        core.part_4,
        core.part_5,
        core.part_6,
    ]
    args = parser.parse_args()
    try:
        logging.info(f"Executing part {args.part}")
        _fn[args.part - 1]()
    except IndexError:
        logging.error(f"Part {args.part} is not implemented")
        return

if __name__ == "__main__":
    main()