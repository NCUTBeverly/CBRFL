import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


class Contribution:

    def __init__(self, filename):
        self._filename = filename
        if os.path.exists(filename):
            self._df = pd.read_csv(filename, index_col=0)
        else:
            self._df = pd.DataFrame()

    def add_column(self, head: str, col: dict):
        if head in self._df:
            self._df.pop(head)
        self._df[head] = pd.Series(col)

    def save(self):
        self._df.to_csv(self._filename)


# if __name__ == "__main__":
#     c = Contribution("contrib.csv")
#     c.add_column("fltrust3", {i: 2. for i in range(20)})
#     c.add_column("fltrust1", {i: 3. for i in range(20)})
#     c.flush()
