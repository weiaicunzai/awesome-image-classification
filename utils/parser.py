import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data")
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--model_path', default='log')

    args = parser.parse_args()
    return args