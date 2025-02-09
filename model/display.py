import numpy as np
import os
import numbers

class Display:
    def __init__(self):
        self.header_displayed = False
        self.attributes = {}
        self.output = ''
        self.format = None
        self.width = None
        self.count = 0
        self.display_top = -1

    def add_attributes(self, name, value, format=True, width=3):
        self.format = format
        self.width = width
        self.attributes[name] = value


    def do(self, algorithm):
        self.__count_up()
        self._do(algorithm)
        if not self.header_displayed:
            self.__make_row(self.attributes.keys())
            self.header_displayed = True
            self.output += '\n'
            header_length = len(self.output)
            lines = self.__get_horizontal_line(header_length)
            self.output += lines
            self.output = lines + self.output

            
        self.__make_row(self.attributes.values())
        self.__display()
        self.__clear()


    ## Protected Methods ##
    def _do(self, algorithm):
        pass

    ## Private Methods ##
    def __make_row(self, columns):
        for col in columns:
            if not isinstance(col, (numbers.Number, str)):
                self.output += '|{}  '
            else:
                self.output += '|{:>10}  '
        if self.format and self.header_displayed:
            columns = list(map(self.__format_number, columns))

        self.output = self.output.format(*columns)

    def __clear(self):
        self.output = ''
        self.attributes.clear()

    def __display(self):
        print(self.output)

    def __format_number(self, number):
        if isinstance(number, (numbers.Number)):
            return round(number, self.width)
        np.set_printoptions(suppress=True, precision=self.width)
        return number

    def __get_horizontal_line(self, header_length):
        lines = ''
        for i in range(header_length):
            lines += '='
        lines += '\n'
        return lines

    def __count_up(self):
        if self.display_top == -1:
            return
        self.count += 1
        if self.count % self.display_top == 0:
            self.header_displayed = False
            os.system('clear')
        