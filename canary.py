from canarymodule import Canary

def main():
    model = Canary()
    filename = input("hello please input the data file name")
    model.predict(filename)
    


