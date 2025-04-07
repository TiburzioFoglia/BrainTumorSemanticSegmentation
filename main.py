import argparse
from Dataset.coco_to_masks import make_masks
from Dataset.resize_images import resize_all
from unet import run_model

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data management command
    data_parser = subparsers.add_parser("data", help="Manage dataset")
    data_parser.add_argument("--masks", action="store_true", help="Create the mask datasets from the annotations")
    data_parser.add_argument("--resize", action="store_true", help="Resize the datasets")

    # Model start command
    model_parser = subparsers.add_parser("train", help="Train the model")
    model_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    model_parser.add_argument("--existing-data", type=bool, default=False, help="Train from existing model")

    model_parser = subparsers.add_parser("test", help="Test the model")
    model_parser.add_argument("--n-elements", type=int, default=5, help="Number of elements to test")

    args = parser.parse_args()

    if args.command == "data":
        if args.masks:
            make_masks()
        if args.resize:
            resize_all()

    elif args.command == "train":
        run_model(training = True, epochs=args.epochs, existing_data=args.existing_data)

    elif args.command == "test":
        run_model(training=False, n_training_elements = args.n_elements)

if __name__ == "__main__":
    main()
