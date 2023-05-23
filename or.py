from neuronlinear.model import Perceptron, prepare_data

import pandas as pd
import logging
import os


logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(logs_dir, "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode='a'
)


def main(data, model_name, eta=0.1, epochs=10, model_dir="Perceptron Models"):

    logging.info(f"This is the raw dataset : \n{data}")
    model = Perceptron(eta, epochs)

    x, y = prepare_data(data, target_col="y")

    model.fit(x, y)

    logging.info(f"Prediction for > 0, 1 is {model.predict([0, 1])}")

    logging.info(f"Total loss is {model.total_loss}")

    #saving model

    model.save(model_name, model_dir)


if __name__  == "__main__":
    data = {"x1":[0, 0, 1, 1], "x2" : [0, 1, 0, 1], "y" : [0, 1, 1, 1]}
    try:
        logging.info("<<<<<<<<<<<< started training >>>>>>>>>>>>>>>>")
        main(data, model_name="or.model")
        logging.info("<<<<<<<<<<<< done training >>>>>>>>>>>>>>")
    except Exception as e:
        logging.exception(e)