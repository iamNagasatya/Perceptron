from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd


def main(data, model_name="or.model", eta=0.1, epochs=10, model_dir="Perceptron Models"):
    df = pd.DataFrame(data)
    model = Perceptron(eta, epochs)

    x, y = prepare_data(df, target_col="y")

    model.fit(x, y)

    print(f"Prediction for > 0, 1 is {model.predict([0, 1])}")

    print(f"Total loss is {model.total_loss}")

    #saving model

    model.save(model_name, model_dir)


if __name__  == "__main__":
    data = {"x1":[0, 0, 1, 1], "x2" : [0, 1, 0, 1], "y" : [0, 0, 0, 1]}
    main(data)
