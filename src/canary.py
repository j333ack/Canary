from canarymodule import Canary

def main():
    model = Canary()
    filename = input("hello please input the data file name: ")
    model.prediction(filename)

main()
    


