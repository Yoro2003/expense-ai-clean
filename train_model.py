import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import os
# -*- coding: utf-8 -*-

# Load dummy data
data = pd.read_csv("data/training_data.csv")

X = data[["amount", "category", "type"]].values
y = data["label"].values
y_encoded = to_categorical(y)

model = Sequential()
model.add(Dense(16, input_dim=3, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(2, activation="softmax"))  # 2 classes: essentiel / non_essentiel

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y_encoded, epochs=50, batch_size=4, verbose=1)

os.makedirs("model", exist_ok=True)
model.save("model/expense_model.h5")
print("✅ Modèle entraîné et sauvegardé")
