import math
from torchvision import datasets, transforms #type:ignore
import json
from random import random
from math import e


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

test_data = datasets.MNIST(
    root="./test_data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print(len(test_data))

# print(dataset)
# img , label = dataset[0]
# img_flat = img.flatten().tolist()

# print(img_flat)
# print(label)
def run_layer(data,weights):
    output = []
    for n in range(len(weights)):
        bias = weights[n][0][0]
        result = 0
        for w in range(0,len(data)):
            result += (weights[n][1][w]*data[w])
        result += bias
        result = ReLU(result)
        output.append(result)
    print(len(output))
    return output 

def init_randomWeights(neurons = 128, input_size = 784):
    # weihts = [[[bias],[weights]]neuron]every neuron
    weights = [[[0],[None]*784] for _ in range(neurons)]
    for n in range(len(weights)):
        for w in range(0,784):
            weights[n][1][w] = (random()/5) - 0.1
    return weights

def ReLU(num):
    if num > 0:
        return num
    else:
        return 0
    
def softmax(arr):
    sum = 0
    for i in arr:
        sum += e ** i
    for i in range(len(arr)):
        arr[i] = e ** arr[i]/sum
    return arr


def calculate_loss(right_answer, output): #Cross-Entropy Loss: L = -∑ yᵢ log(ŷᵢ)
    epsilon = 1e-15
    value = max(epsilon, min(output[right_answer], 1 - epsilon))
    return -math.log(value)


def run_all_layers(img):
    hidden_layer_result = run_layer(img, weights=weights)
    print(hidden_layer_result)
    output = run_layer(hidden_layer_result,weights=output_weights)
    print(output)
    print(softmax(output))
    return softmax(output)
# weights = init_randomWeights()
# output_weights = init_randomWeights(10,128)






with open("hidden_layer.json" ,"w") as  f:
    json.dump(weights, f)

with open("output_layer.json" ,"w") as  f:
    json.dump(output_weights, f)    