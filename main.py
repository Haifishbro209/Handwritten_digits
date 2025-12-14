from torchvision import datasets, transforms #type:ignore
import json

with open("weights.json") as f:
    weights = json.load(f)

print(weights)

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print(dataset)
img , label = dataset[0]
print(img)
print(label)

weights = [6,7,9]
with open("weights.json" ,"w") as  f:
    json.dump(weights, f)