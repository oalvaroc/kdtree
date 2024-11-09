from kdtree import KDTree

if __name__ == "__main__":
    tree = KDTree(2)
    tree.insert([30, 40])
    tree.insert([5, 25])
    tree.insert([10, 12])
    tree.insert([70, 70])
    tree.insert([33, 30])
    tree.insert([35, 45])

    print("TRAVERSE:")
    tree.traverse(print)

    point = [70, 70]
    tree.remove(point)
    print(f"\nREMOVE {point}:")
    tree.traverse(print)

    print("\nNEAREST NEIGHBOUR")
    print(tree.nearest_neighbour([12, 20]))
