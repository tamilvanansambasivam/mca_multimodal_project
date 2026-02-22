import numpy as np
import pandas as pd

np.random.seed(42)

orientations = np.arange(0, 360, 30)

def generate_sample():
    F = np.random.uniform(0, 8)      # Force
    C = np.random.uniform(0, 15)     # Curvature
    O = np.random.choice(orientations)
    return F, C, O

def sensor_model(F, C, O):
    noise = np.random.normal(0, 0.05, 5)

    S1 = np.sin(np.radians(O)) + 0.1*C + 0.2*F + noise[0]
    S2 = np.cos(np.radians(O)) + 0.05*C**2 + 0.1*F + noise[1]
    S3 = 0.3*C + 0.2*np.sin(F) + noise[2]
    S4 = 0.2*F + 0.1*C*np.cos(np.radians(O)) + noise[3]
    S5 = 0.05*F*C + 0.1*np.sin(np.radians(O)) + noise[4]

    return [S1, S2, S3, S4, S5]

data = []

for _ in range(20000):
    F, C, O = generate_sample()
    signals = sensor_model(F, C, O)
    data.append(signals + [F, C, O])

columns = ["S1","S2","S3","S4","S5","Force","Curvature","Orientation"]
df = pd.DataFrame(data, columns=columns)

df.to_csv("dataset.csv", index=False)
print("Dataset created successfully.")