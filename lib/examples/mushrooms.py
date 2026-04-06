import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_mushrooms(ruleset = lambda i, j: i + j > 1, size_shrooms = 2000):
    rng = np.random.default_rng(seed=42)
    mushrooms = []
    poisonous = []
    for _ in range(size_shrooms):
        redness = rng.random()
        dots = rng.random()
        is_poisonous = ruleset(redness, dots)
        mushrooms.append([redness, dots])
        poisonous.append([1 if is_poisonous else 0])
    return np.array(mushrooms), np.array(poisonous)

def plot_mushrooms(X, y):
    df = pd.DataFrame({
    "redness(%)": X[:, 0],
    "dots(%)": X[:, 1],
    "poisonous": y[:, 0]
    })

    poisonous = df[df["poisonous"] >= 0.5]
    not_poisonous = df[df["poisonous"] < 0.5]
    poisonous_x = poisonous["redness(%)"]
    poisonous_y = poisonous["dots(%)"]
    not_poisonous_x = not_poisonous["redness(%)"]
    not_poisonous_y = not_poisonous["dots(%)"]

    plt.scatter(poisonous_x, poisonous_y, c="pink")
    plt.scatter(not_poisonous_x, not_poisonous_y, c="lightblue")

    # Add labels and a title
    plt.xlabel('Dots Coverage (%)')
    plt.ylabel('Red intensity (%)')
    plt.title('Poisonous Mushrooms')
    plt.show()

def plot_mushroom_comparison(X1, y1, X2, y2):
    df1 = pd.DataFrame({
    "redness(%)": X1[:, 0],
    "dots(%)": X1[:, 1],
    "poisonous": y1[:, 0]
    })

    df2 = pd.DataFrame({
    "redness(%)": X2[:, 0],
    "dots(%)": X2[:, 1],
    "poisonous": y2[:, 0]
    })
    
    agree = df1[(df1["poisonous"] >= 0.5) == (df2["poisonous"]>= 0.5)]

    agree_poisonous = agree[agree["poisonous"] >= 0.5]
    agree_poisonous_x = agree_poisonous["redness(%)"]
    agree_poisonous_y = agree_poisonous["dots(%)"]

    agree_not_poisonous = agree[agree["poisonous"] < 0.5]
    agree_not_poisonous_x = agree_not_poisonous["redness(%)"]
    agree_not_poisonous_y = agree_not_poisonous["dots(%)"]

    disagree = df1[(df1["poisonous"] >= 0.5) != (df2["poisonous"]>= 0.5)]
    
    disagree_poisonous = disagree[disagree["poisonous"] >= 0.5]
    disagree_poisonous_x = disagree_poisonous["redness(%)"]
    disagree_poisonous_y = disagree_poisonous["dots(%)"]

    disagree_not_poisonous = disagree[disagree["poisonous"] < 0.5]
    disagree_not_poisonous_x = disagree_not_poisonous["redness(%)"]
    disagree_not_poisonous_y = disagree_not_poisonous["dots(%)"]

    plt.scatter(disagree_poisonous_x, disagree_poisonous_y, c="darkgreen")
    plt.scatter(disagree_not_poisonous_x, disagree_not_poisonous_y, c="darkblue")

    plt.scatter(agree_poisonous_x, agree_poisonous_y, c="lightgreen")
    plt.scatter(agree_not_poisonous_x, agree_not_poisonous_y, c="lightblue")

    # Add labels and a title
    plt.xlabel('Dots Coverage (%)')
    plt.ylabel('Red intensity (%)')
    plt.title('Poisonous Mushrooms')