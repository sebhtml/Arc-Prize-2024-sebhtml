
class QLearningAction:
    def __init__(self, row, col, cell_value):
        self.__row = row
        self.__col = col
        self.__cell_value = cell_value

    def row(self) -> int:
        return self.__row

    def col(self) -> int:
        return self.__col

    def cell_value(self) -> int:
        return self.__cell_value
