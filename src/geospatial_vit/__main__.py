"""
Main entry point for Geospatial ViT
"""
import argparse
import sys

def train_command(args):
    """Handle train command"""
    from .train_real import main
    main(args.config)

def main():
    parser = argparse.ArgumentParser(description='Geospatial Vision Transformer')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Version command
    parser.add_argument('--version', action='store_true', help='Show version')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to config file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    predict_parser.add_argument('--image', type=str, required=True,
                               help='Path to input image')
    
    args = parser.parse_args()
    
    if args.version:
        from . import __version__
        print(f"Geospatial ViT v{__version__}")
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        print("Predict command - To be implemented")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
