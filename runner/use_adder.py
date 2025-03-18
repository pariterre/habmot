import habmot
import numpy as np


def main():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = habmot.adder(a, b)
    print(
        f"The result of adding {a} and {b} is {result}.\n"
        f'This is computed using habmot.adder from version "{habmot.__version__}"'
    )


if __name__ == "__main__":
    main()
