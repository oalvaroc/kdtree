"""
KD-Tree
=======

A data structure to organize points in a K-dimensional space.
"""

from collections import deque
from collections.abc import Callable
from typing import Self, Optional, Literal
import numpy as np

import graphviz
import math

def euclid2(a: list | int, b: list | int) -> int:
    """Compute squared euclidean distance.

    The squared euclidean distance is defined as:

    .. math::
        d(a, b) = \\sum_{i=1}^{k} (a_{i} - b_{i})^2

    Args:
        a: Point in k-dimensional space.
        b: Point in k-dimensional space.

    Returns:
        int: The squared euclidean distance between `a` and `b`
    # """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b)


class KDNode:
    """A k-dimensional node from a kd-tree.

    Attributes:
        point (list): Point in the k-dimensional space that this node represents
        left (KDNode, optional): Left child node.
        right (KDNode, optional): Right child node.
    """

    def __init__(
        self, point=[], left: Optional[Self] = None, right: Optional[Self] = None
    ):
        self.point = np.asarray(point)
        self.left = left
        self.right = right

    def is_leaf(self):
        """Check if node is a leaf node.

        A leaf node is a node without any child.

        Returns:
            bool: `True` if node is leaf or `False` otherwise.
        """
        return self.left is None and self.right is None

    def __len__(self):
        return len(self.point)

    def __repr__(self):
        right = None if self.right is None else hex(id(self.right))
        left = None if self.left is None else hex(id(self.left))
        return (
            f"{KDNode.__name__}(values={list(self.point)}, left={left}, right={right})"
        )


class KDTree:
    """A k-dimensional tree.

    A kd-tree is a binary tree that partitions a k-dimensional space to organize
    points in that space.

    Attributes:
        ndim (int): Number of dimensions of points stored in the tree.
    """

    def __init__(self, ndim: int):
        self._ndim = ndim
        self._root = None

    @staticmethod
    def from_points(points: list, ndim: int, method: str = Literal["sort", "naive", "subset"], subsize=10):
        points = np.asarray(points)
        tree = KDTree(ndim)

        if method == "sort":
            tree._root = KDTree._build_sorted(ndim, 0, points)
        elif method == "subset":
            tree._root = KDTree._build_sorted_subset(ndim, 0, points, subsize)
        elif method == "naive":
            for point in points:
                tree.insert(point)
        return tree
        
    @property
    def ndim(self):
        """Number of dimensions of points stored in the tree."""
        return self._ndim

    def traverse(self, callable: Callable[[KDNode], None]):
        """BFS tree traversal.

        Traverse tree using Breadth-First Search algorithm and invoke
        `callable` for each visited node.

        Args:
            callable (Callable[[KDNode], None]):
                A callable object that is invoked at every visited node.
        """
        queue = deque([self._root])
        visited = set()

        while queue:
            node = queue.popleft()
            visited.add(node)
            callable(node)

            for next_node in [node.left, node.right]:
                if next_node is not None and next_node not in visited:
                    queue.append(next_node)

    def insert(self, point: list):
        """Add `point` to the tree.

        Args:
            point (list): A k-dimensional point to add to the tree.

        Raises:
            ValueError: If `point` is already in the tree.
        """
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        self._root = self.__insert(point, self._root, 0)

    def remove(self, point: list):
        """Remove `point` from the tree.

        Args:
            point (list): Point to search and remove from the tree.

        Raises:
            ValueError: If `point` is not found in the tree.
        """
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        self._root = self.__remove(point, self._root, 0)

    def nearest_neighbour(self, point: list) -> tuple[float, KDNode]:
        """Find nearest neighbour of `point`.

        Args:
            point (list): Point to search neighbour for.

        Returns:
            (float, KDNode): tuple with euclidean distance and the node that is
            the closest to `point`.
        """
        best_dist = float("inf")
        best_node = None

        def nn(node: KDNode, dim: int):
            nonlocal best_dist
            nonlocal best_node

            if node is None:
                return

            next_dim = (dim + 1) % self.ndim

            # traverse tree until a leaf node is found and keep track
            # of which subtree we visit: left (0) or right (1).
            if point[dim] <= node.point[dim]:
                nn(node.left, next_dim)
                visited = 0
            else:
                nn(node.right, next_dim)
                visited = 1

            # update best distance if we are currently in a node that is
            # closer to the target `point`
            d = euclid2(point, node.point)
            if d < best_dist:
                best_dist = d
                best_node = node

            # check if need to explore the other subtree. This is done
            # when the hypersphere centered at `point` crosses another
            # neighbouring hyperplane
            if euclid2(point[dim], node.point[dim]) < best_dist:
                nn(node.right if visited == 0 else node.left, next_dim)

        nn(self._root, 0)
        return math.sqrt(best_dist), best_node

    @staticmethod
    def _build_sorted(ndim: int, dim: int, points: np.ndarray):
        if len(points) == 0:
            return None
    
        sorted_points = points[points[:, dim].argsort()]
        median = len(sorted_points) // 2

        next_dim = (dim + 1) % ndim
        node = KDNode(sorted_points[median])
        node.left = KDTree._build_sorted(ndim, next_dim, sorted_points[ : median])
        node.right = KDTree._build_sorted(ndim, next_dim, sorted_points[median + 1 : ])

        return node

    @staticmethod
    def _build_sorted_subset(ndim: int, dim: int, points: np.ndarray, subset_size=10):
        if len(points) == 0:
            return None
    
        # subset = np.random.permutation(points)[ : subset_size]
        # subset = subset[subset[:, dim].argsort()]
        subset = points[np.random.choice(len(points), size=subset_size, replace=False)]
        subset = subset[subset[:, dim].argsort()]

        median = len(subset) // 2
        pivot = points.tolist().index(subset[median].tolist())

        next_dim = (dim + 1) % ndim
        node = KDNode(points[pivot])
        node.left = KDTree._build_sorted(ndim, next_dim, points[ : pivot])
        node.right = KDTree._build_sorted(ndim, next_dim, subset[pivot + 1 : ])

        return node
    

    def __insert(self, point: np.ndarray, node: KDNode, dim: int):
        if node is None:
            node = KDNode(point)
        elif np.array_equal(node.point, point):
            raise ValueError(f"Point '{point}' already exists")
        elif point[dim] <= node.point[dim]:
            node.left = self.__insert(point, node.left, (dim + 1) % self.ndim)
        else:
            node.right = self.__insert(point, node.right, (dim + 1) % self.ndim)
        return node

    def __remove(self, point: list, node: KDNode, dim: int):
        if node is None:
            raise ValueError(f"Point '{point}' not found")

        next_dim = (dim + 1) % self.ndim

        if point == node.point:
            # we found point
            if node.right is not None:
                # replace node with its immediate successor in its
                # right subtree, with respect to dimension `dim`
                node.point = self.__minimum(node.right, dim, next_dim)
                node.right = self.__remove(node.point, node.right, next_dim)
            elif node.left is not None:
                # node has only a left subtree, so we replace node with the
                # minimum node in the left (with respect to `dim`) and move
                # the left subtree to the right.
                node.point = self.__minimum(node.left, dim, next_dim)
                node.right = self.__remove(node.point, node.left, next_dim)
                node.left = None
            else:
                # leaf node
                node = None
        elif point[dim] <= node.point[dim]:
            # search for `point` on the left subtree
            node.left = self.__remove(point, node.left, next_dim)
        else:
            # search for `point` on the right subtree
            node.right = self.__remove(point, node.right, next_dim)

        return node

    def __minimum(self, node: KDNode, dim: int, current_dim: int) -> list | None:
        if node is None:
            return None

        next_dim = (current_dim + 1) % self.ndim
        if current_dim == dim:
            # we are in the dimension we're comparing
            if node.left is None:
                # left subtree doesn't exist, so the minimum node is the
                # current one
                return node.point
            else:
                return self.__minimum(node.left, dim, next_dim)
        else:
            # the current tree level is not the dimension we're are
            # comparing to, so we need to search for the minimum in
            # both left and right subtrees.
            points = [
                node.point,
                self.__minimum(node.left, dim, next_dim),
                self.__minimum(node.right, dim, next_dim),
            ]
            min_point = points[0]
            for p in points:
                if p is not None and p[dim] < min_point[dim]:
                    min_point = p
            return min_point


def plot_tree(tree: KDTree, filename: str):
    if tree is None or tree._root is None:
        return

    dot = graphviz.Digraph("kd-tree")
    dot.node(f"{id(tree._root)}", f"{tree._root.point}")

    def _plot(node: KDNode):
        if node is None:
            return
    
        if node.left:
            dot.node(f"{id(node.left)}", f"{node.left.point}")
            dot.edge(f"{id(node)}", f"{id(node.left)}")

        if node.right:
            dot.node(f"{id(node.right)}", f"{node.right.point}")
            dot.edge(f"{id(node)}", f"{id(node.right)}")

        _plot(node.left)
        _plot(node.right)

    _plot(tree._root)
    dot.render(filename, format="png")