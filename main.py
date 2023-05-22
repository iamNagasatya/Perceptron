from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd

model = Perceptron(eta=0.1, epochs=10)

df = pd.DataFrame({"x1":[0, 0, 1, 1], "x2" : [0, 1, 0, 1], "y" : [0, 0, 0, 1]})

x, y = prepare_data(df, target_col="y")

model.fit(x, y)

print(f"Prediction for > 0, 1 is {model.predict([0, 1])}")

print(f"Total loss is {model.total_loss}")

#saving model

model.save("perceptron_or_gate.model", "Perceptron Models")

