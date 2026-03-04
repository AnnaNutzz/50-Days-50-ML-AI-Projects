import math

def entropy(probs):
    """
    Computes entropy given a list of probabilities.
    """
    h = 0
    for p in probs:
        if p == 0:
            continue
        h -= p * math.log2(p)
    return round(h, 4)


def gini(probs):
    """
    Computes gini impurity given a list of probabilities.
    """
    g = 1
    for p in probs:
        g -= p ** 2
    return round(g, 4)

def counts_to_probs(counts):
    total = sum(counts)
    if total == 0:
        raise ValueError("Total count cannot be zero.")
    return [c / total for c in counts]


def calculate_node_impurity():
    print("Decision Tree – Entropy & Gini Calculator")
    print("---------------------------------------")

    c1 = int(input("Enter count for Class 1: "))
    c2 = int(input("Enter count for Class 2: "))

    probs = counts_to_probs([c1, c2])

    print(f"\nProbabilities:")
    print(f"p1 = {probs[0]:.3f}, p2 = {probs[1]:.3f}")

    print("\nChoose impurity metric:")
    print("1. Entropy")
    print("2. Gini Impurity")

    choice = input("Enter choice (1/2): ")

    if choice == "1":
        h = entropy(probs)
        print(f"\nEntropy = {h}")
    elif choice == "2":
        g = gini(probs)
        print(f"\nGini Impurity = {g}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    calculate_node_impurity()