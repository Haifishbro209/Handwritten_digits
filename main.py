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

def calculate_cost(right_answer, output):
    cost = 0
    for i in range(len(output)):
        if i == right_answer:
            cost += (output[i] - 1)**2
        else:
            cost += output[i]**2
    return cost



def run_all_layers(img):
    hidden_layer_result = run_layer(img, weights=weights)
    print(hidden_layer_result)
    output = run_layer(hidden_layer_result,weights=output_weights)
    print(output)
    print(softmax(output))
    return softmax(output)
# weights = init_randomWeights()
# output_weights = init_randomWeights(10,128)


def update_weights(weights,img_flat, label, learning_rate = 0.01):
    for neuron in range(len(weights)):
        for w in range(len(weights[neuron][1])):
            weight = weights[neuron][1][w]
            
            weights[neuron][1][w] -= 0.001
            output = run_all_layers(img_flat)
            error_small_weight = calculate_cost(label,output)
            weights[neuron][1][w] += 0.002
            output = run_all_layers(img_flat)
            error_big_weight = calculate_cost(label,output)

            steigung = (error_big_weight - error_small_weight ) / 0.002

            weights[neuron][1][w] = weight - (steigung * learning_rate)



def train_one_epoch():
    for i in range(len(dataset)):#only half of the trainingsdata
        img , label = dataset[i]
        img_flat = img.flatten().tolist()

        update_weights(weights,img_flat,label) #run hidden layer
        update_weights(output_weights,img_flat,label) #run outpuit layer


with open("hidden_layer.json" ,"w") as  f:
    json.dump(weights, f)

with open("output_layer.json" ,"w") as  f:
    json.dump(output_weights, f)    