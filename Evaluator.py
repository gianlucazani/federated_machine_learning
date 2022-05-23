import matplotlib.pyplot as plt
import pandas


class Evaluator:
    def __init__(self, files):
        self.files = files

    def run(self):
        for file in self.files:
            figure, axis = plt.subplots(2, 2)
            data = pandas.read_csv(file)

            axis[0, 0].plot(data["communication_round"], data["accuracy"], color="blue")
            axis[0, 0].set_title("Client1")

            # For Cosine Function
            # axis[0, 1].plot(X, Y2)
            # axis[0, 1].set_title("Cosine Function")
            #
            # data.plot(x="communication_round", y="accuracy", color="blue")
            # data.plot(x="communication_round", y="loss", color="red")
        plt.show()

files = ["evaluation_log_1.csv"]

evaluator = Evaluator(files)
evaluator.run()
