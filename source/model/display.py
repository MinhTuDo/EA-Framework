import numpy as np

class Display:
    def __init__(self):
        self.header_displayed = False
        self.attributes = {}
        self.output = ''
        self.format = None
        self.width = None

    def add_attributes(self, name, value, format=True, width=3):
        self.format = format
        self.width = width
        self.attributes[name] = value


    def do(self, algorithm):
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
            if not isinstance(col, (int, float, str)):
                self.output += '|{}\t'
            else:
                self.output += '|{:^12}'
        if self.format and self.header_displayed:
            columns = list(map(self.__format_number, columns))

        self.output = self.output.format(*columns)

    def __clear(self):
        self.output = ''
        self.attributes.clear()

    def __display(self):
        print(self.output)

    def __format_number(self, number):
        if isinstance(number, (int, float, complex)):
            return round(number, self.width)
        np.set_printoptions(suppress=True, precision=self.width)
        return number

    def __get_horizontal_line(self, header_length):
        lines = ''
        for i in range(header_length):
            lines += '='
        lines += '\n'
        return lines
        