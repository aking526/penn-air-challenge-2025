import argparse
import corev2 as core
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
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed video output to MP4 (parts 2 and 3)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output MP4 path; defaults per input video"
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
        if args.part == 2:
            core.part_2(save=args.save, output_path=args.out)
        elif args.part == 3:
            core.part_3(save=args.save, output_path=args.out)
        else:
            _fn[args.part - 1]()
    except IndexError:
        logging.error(f"Part {args.part} is not implemented")
        return

if __name__ == "__main__":
    main()