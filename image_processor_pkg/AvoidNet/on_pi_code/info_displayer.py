import os

class DynamicDictDisplay_bare:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.strings = []

    def update_display(self):
        self.strings = []
        # clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')
        for key, value in self.data_dict.items():
            self.strings.append(f"{key:>10} : {value}")
        print("=== variables ===")
        print(*self.strings, sep="\n")
        print ("=================")

    def update_dict(self, data_dict):
        self.data_dict = data_dict
        self.update_display()

    def cleanup(self):
        pass
