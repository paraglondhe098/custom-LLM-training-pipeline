import argparse
from app import App


def main():
    parser = argparse.ArgumentParser(description="Run various pipelines in the App.")
    parser.add_argument(
        "action",
        choices=["ingest", "tokenize", "train", "generate", "run"],
        help="Specify the pipeline to run."
    )
    args = parser.parse_args()

    app = App()

    if args.action == "ingest":
        app.ingest_data()
    elif args.action == "tokenize":
        app.tokenize_text()
    elif args.action == "train":
        app.train_model()
    elif args.action == "generate":
        app.generate_responses()
    elif args.action == "run":
        app.run_pipeline()


if __name__ == "__main__":
    main()
