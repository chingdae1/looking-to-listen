import argparse
import yaml
from solver import Solver


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--num_of_face', type=int, default=2)
    # parser.add_argument('--vgg_face_path', type=str, default='./asset/vgg_face_dag.pth')
    #
    # parser.add_argument('--lr', type=float, default=0.00003)
    # parser.add_argument('--epoch', type=int, default=100)
    #
    # parser.add_argument('--data_dir', type=str, default='./data_toy')
    # parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--num_workers', type=int, default=4)
    #
    # parser.add_argument('--model_name', type=str, default='first_trial')
    # parser.add_argument('--val_every', type=int, default=1)
    # parser.add_argument('--sample_for', type=int, default=10)
    # parser.add_argument('--val_sample_dir', type=str, default='./val_sample')
    # parser.add_argument('--save_dir', type=str, default='./saved')

    parser.add_argument('--config_path', type=str, default='./options/baseline.yaml')

    args = parser.parse_args()

    with open(args.config_path, 'r') as config
        config = yaml.load(config.read())
    solver = Solver(config)
    solver.fit()

if __name__ == '__main__':
    main()
