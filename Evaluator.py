import matplotlib.pyplot as plt
import pandas


class Evaluator:
    def __init__(self, files):
        self.files = files

    def run(self):

        # Plot losses and accuracies of each client
        for file in self.files:
            data = pandas.read_csv(file)
            data.plot(x="communication_round", y="accuracy", color="blue")
            data.plot(x="communication_round", y="loss", color="red")

        plt.show()


files = ["evaluation_log_1.csv", "evaluation_log_2.csv", "evaluation_log_3.csv", "evaluation_log_4.csv", "evaluation_log_5.csv"]

evaluator = Evaluator(files)
evaluator.run()
