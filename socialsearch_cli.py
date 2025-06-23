import argparse
from search_assistant import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multisource social/web search assistant",
        prog="vociro",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Currently we only expose 'init' which launches the interactive assistant.
    sub.add_parser("init", help="Start an interactive research session")

    args = parser.parse_args()

    if args.command == "init":
        run()


if __name__ == "__main__":  # pragma: no cover
    main() 