from torchvision import datasets, transforms #type:ignore
import json
from random import random


with open("hidden_layer.json") as f:
    weights = json.load(f) #weights and biases

with open("output_layer.json") as f:
    output_weights = json.load(f) 

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print(dataset)
img , label = dataset[0]
img_flat = img.flatten().tolist()

print(img_flat)
print(label)

def run_hidden_layer(data):#data is a 784 long arr
    output = []
    for n in range(len(weights)):
        bias = weights[n][0][0]
        result = 0
        for w in range(0,784):
            result += (weights[n][1][w]*data[w])
        #do sth
        result += bias 
        output.append(result)
    print(len(output))
    return output

run_hidden_layer(img_flat)

def init_randomWeights(neurons = 128, input_size = 784):
    # weihts = [[[bias],[weights]]neuron]every neuron
    weights = [[[0],[None]*784] for _ in range(neurons)]
    for n in range(len(weights)):
        for w in range(0,784):
            weights[n][1][w] = (random()/5) - 0.1
    return weights

#weights = init_randomWeights()

output_weights = init_randomWeights(10,128)

with open("hidden_layer.json" ,"w") as  f:
    json.dump(weights, f)

with open("output_layer.json" ,"w") as  f:
    json.dump(output_weights, f)    