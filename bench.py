from kdtree import KDTree
import tqdm

import matplotlib.pyplot as plt
import numpy as np

import argparse
import time

# Benchmarks:
# - number of dimensions vs. search time
# - number of nodes vs. search time
#    - check how tree balancing impacts search
# - sort elements on creation vs. search time
# - get median from small subset of elements
#    - what is a good size of the subset?
# - naive search vs kd-tree


def make_point(dim: int, dim_range: tuple):
    return np.random.randint(*dim_range, size=dim)


def make_tree(dim: int, nodes: int, dim_range: tuple, method: str, worst_case=False, subsize=10):
    if worst_case:
        points = np.arange(nodes * dim).reshape((nodes, dim))
    else:
        points = np.random.randint(*dim_range, size=(nodes, dim))
    return KDTree.from_points(points, dim, method=method, subsize=subsize)


def lineplot(
    x, y, title: str, xlabel: str = None, ylabel: str = None, logx: bool = False, filename: str = None
):
    fig, ax = plt.subplots()
    if x:
        ax.plot(x, y)
    else:
        ax.plot(y)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")

    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    plt.close()


class Bench:
    def __init__(self, dim: int, nodes: int, method: str, worst: bool = False):
        self.dim = dim
        self.nodes = nodes
        self.method = method
        self.worst = worst

    def __call__(self):
        pass


class DimBench(Bench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._point_range = (0, 100000000)

    def __call__(self):
        x = []
        y = []

        for dim in tqdm.tqdm(np.logspace(1, np.log2(self.dim), base=2, dtype=int)):
            tree = make_tree(
                dim, self.nodes, self._point_range, self.method, self.worst
            )

            searches = 10
            point = make_point(dim, self._point_range)

            start = time.perf_counter()
            for _ in range(searches):
                tree.nearest_neighbour(point)
            delta = time.perf_counter() - start

            x.append(dim)
            y.append(delta / searches)

        return x, y


class NodesBench(Bench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._point_range = (0, 100000)

    def __call__(self):
        x = []
        y = []

        for nodes in tqdm.tqdm(
            np.logspace(1, np.log2(self.nodes), 100, base=2, dtype=int)
        ):
            tree = make_tree(
                self.dim, nodes, self._point_range, self.method, self.worst
            )

            searches = 10
            point = make_point(self.dim, self._point_range)

            start = time.perf_counter()
            for _ in range(searches):
                tree.nearest_neighbour(point)
            delta = time.perf_counter() - start

            x.append(nodes)
            y.append(delta / searches)

        return x, y


class MedianBench(Bench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._point_range = (0, 100000)

    def __call__(self):
        x = []
        y = []

        # for nodes in tqdm.tqdm(
        #     np.logspace(1, np.log2(self.nodes), 100, base=2, dtype=int)
        # ):
        #     subx = []
        #     suby = []

        for subsize in tqdm.tqdm(np.logspace(1, np.log2(self.nodes // 2), base=2, dtype=int)):
            # print(f"nodes={nodes}  subsize={subsize}")
            tree = make_tree(
                self.dim, self.nodes, self._point_range, "subset", self.worst, subsize
            )

            searches = 10
            point = make_point(self.dim, self._point_range)

            start = time.perf_counter()
            for _ in range(searches):
                tree.nearest_neighbour(point)
            delta = time.perf_counter() - start

            # subx.append(subsize)
            # suby.append(delta / searches)

            x.append(subsize)
            y.append(delta / searches)

        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KD-Tree benchmark tool")
    parser.add_argument("bench", choices=("dim", "nodes", "median"))
    parser.add_argument(
        "-k", "--dimensions", type=int, default=2, help="Maximum number of dimensions"
    )
    parser.add_argument(
        "-n",
        "--nodes",
        type=int,
        default=1000,
        help="Maximum number of nodes to add to the tree",
    )
    parser.add_argument(
        "--creation-method",
        choices=("sort", "subset", "naive"),
        default="sort",
        help=(
            "Creation method to use. With 'sort', elements are sorted with respect to the current dimension "
            "and then the median is selected. The 'subset' option works like 'sort', but sorting is done on "
            "a subset of elements. With 'naive' elements are inserted in the order they are given."
        ),
    )
    parser.add_argument(
        "--worst",
        action="store_true",
        default=False,
        help="Produce a sorted sequence of points to trigger the worst case: a completely unbalanced tree",
    )

    args = parser.parse_args()
    bench_args = (args.dimensions, args.nodes, args.creation_method, args.worst)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    if args.bench == "dim":
        x, y = DimBench(*bench_args)()
        lineplot(
            x,
            y,
            title=f"Dimension vs. Time (s)\n{args.nodes} nodes, create method={args.creation_method}",
            xlabel="# dimensions",
            ylabel="time (s)",
            filename=f"{args.bench}-n{args.nodes}-{args.creation_method}-{timestamp}.png",
        )
    elif args.bench == "nodes":
        x, y = NodesBench(*bench_args)()
        lineplot(
            x,
            y,
            title=f"Nodes vs. Time (s)\n{args.dimensions} dimensions, create method={args.creation_method}",
            xlabel="# nodes",
            ylabel="time (s)",
            filename=f"{args.bench}-d{args.dimensions}-{args.creation_method}-{timestamp}.png",
        )
    elif args.bench == "median":
        x, y = MedianBench(*bench_args)()
        lineplot(
            x,
            y,
            title=f"Subset size vs. Time (s)\n{args.dimensions} dimensions, {args.nodes} nodes, create method={args.creation_method}",
            xlabel="subset size",
            ylabel="time (s)",
            filename=f"{args.bench}-d{args.dimensions}-n{args.nodes}-{args.creation_method}-{timestamp}.png",
        )
