from torchvision import datasets, transforms #type:ignore
import json
from random import random


with open("weights.json") as f:
    weights = json.load(f) #and biases

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




def init_randomWeights(neurons = 128):
    # weihts = [[[bias],[weights]]neuron]every neuron
    weights = [[[0],[None]*784] for _ in range(neurons)]
    for n in range(len(weights)):
        for w in range(0,784):
            weights[n][1][w] = (random()/5) - 0.1
    return weights

#weights = init_randomWeights()

with open("weights.json" ,"w") as  f:
    json.dump(weights, f)