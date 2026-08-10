import math
from torchvision import datasets, transforms #type:ignore
import json
from random import random
from math import e

global grad_L1_weights, grad_L1_bias, grad_L2_weights, grad_L2_bias, loss
grad_L2_weights = [[0]*128]*10
grad_L2_bias = [0]*10
grad_L1_weights = [[0]*784]*128
grad_L1_bias = [0]*128


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

def forwardpass(img , label):
    #img ist 784 array mit bilddaten
    #L1 hiddenlayer L2 outputlayer
    L1 = run_layer(img, weights=weights)
    print(L1)
    L2 = run_layer(L1,weights=output_weights)
    print(L2)
    output = softmax(L2)
    print(output)
    total_loss = calculate_loss(label,output)
    loss += total_loss
    return output,L1, label, total_loss , img

# weihts = [[[bias],[weights]],[[bias],[weights]]]
#                  neuron1          neuron2

#   x[] -> x[0]*  w[0][0] +...+x[783]*  w[0][783] + bias[0] -> ReLU() = L1[0]    
#       -> x[0]*w[127][0] +...+x[783]*w[127][783] + bias[783] -> ReLU() = L1[127]
#  =>> L1 ##diferent weights 
#   L1[] -> L1[0] * w[0][0] + ... L1[127]*w[0][127]+ bias[0] -> ReLU() = L2[0]
#        -> L1[0] * w[9][0] + ... L1[127]*w[9][127]+ bias[9] -> ReLU() = L2[9]
#   softmax(L2) (-> cross entrophy loss)
#   
'''
Für jedes Output-Neuron i (0 bis 9) und jedes Hidden-Neuron j (0 bis 127):
grad_L2_weights[i][j] = error_output[i] * L1[j]
'''
def backwards(L2, L1, label, x):
    y_true = [0]*10
    y_true[label] = 1


    for i in range(10):
        local_error = L2[i] -  y_true[i]
        grad_L2_bias[i] += local_error
        for j in range(128):
            grad_L2_weights[i][j] +=  L1[j] * local_error

    error_hidden = [0]*128

    for i in range(128):
        if L1[i] ==0:
            error_hidden[i] = 0
        else :
            for n in range(10):
                error_hidden[i] += grad_L2_bias[n] * output_weights[n][1][i]
        grad_L1_bias[i] += error_hidden[i]

        for j in range(784):
            grad_L1_weights[i][j] += x[j] * error_hidden[i]

    #return grad_L1_weights, grad_L1_bias, grad_L2_weights, grad_L2_bias

# weihts = [[[bias],[weights]],[[bias],[weights]]]
#                  neuron1          neuron2



def run(img,label, LR):
    L2,L1, label, total_loss , img = forwardpass(img,label)
    g_L1_weights, g_L1_bias, g_L2_weights, g_L2_bias = backwards(L2,L1, label,img)
    total_loss
    
def run_epoche(size = 1000):
    for i in range(size):
        img, label = dataset[i]




    
def update_network(grad_L1_weights, grad_L1_bias, grad_L2_weights, grad_L2_bias, ts ,LR = 0.2):
    #ts training size
    for n in range(128):
        weights[n][0][0] = weights[n][0][0] - (grad_L1_bias[n]/ts) *LR 
        for i in range(784):
            weights[n][1][i] = weights[n][1][i] - (grad_L1_weights[n][i]/ts) * LR
    for n in range(10):
        output_weights[n][0][0] =output_weights[n][0][0] - (grad_L2_bias[n]/ts) * LR
        for i in range(128):
            output_weights[n][1][i] = output_weights[n][1][i] - (grad_L2_weights[n][i]/ts) *LR








with open("hidden_layer.json" ,"w") as  f:
    json.dump(weights, f)

with open("output_layer.json" ,"w") as  f:
    json.dump(output_weights, f)    