import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "speed": np.random.normal(50, 15, n),  # km/h
    "acceleration": np.random.normal(0, 2, n),  # m/s^2
    "braking": np.random.normal(-1, 2, n),
    "turn_rate": np.random.normal(10, 5, n)  # degree/sec
})

# Label: 1 = Safe, 0 = Unsafe
data["label"] = (
    (data["speed"] < 70) &
    (data["acceleration"] < 3) &
    (data["braking"] > -3) &
    (data["turn_rate"] < 20)
).astype(int)

data.to_csv("data.csv", index=False)

print("Dataset generated!")