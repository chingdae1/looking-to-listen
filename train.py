import argparse
from solver import Solver


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_of_face', type=int, default=1)
    parser.add_argument('--vgg_face_path', type=str, default='./asset/vgg_face_dag.pth')

    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--epoch', type=int, default=15)

    parser.add_argument('--data_dir', type=str, default='./data_toy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    config = parser.parse_args()
    solver = Solver(config)
    solver.fit()

if __name__ == '__main__':
    main()
