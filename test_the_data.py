import matplotlib.pyplot as plt
import numpy as np
from Numerical_Differentiation import dataset

# 1. Beispielbilder anzeigen
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img, label = dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 2. Klassenverteilung
labels = [label for _, label in dataset]
plt.hist(labels, bins=10, edgecolor='black')
plt.xlabel('Ziffer')
plt.ylabel('Häufigkeit')
plt.title('Klassenverteilung im MNIST-Trainingsdatensatz')
plt.show()

# 3. Pixel-Statistiken
all_pixels = []
for img, _ in dataset:
    all_pixels.extend(img.flatten().tolist())
print(f"Min: {min(all_pixels)}, Max: {max(all_pixels)}, Mean: {np.mean(all_pixels)}")