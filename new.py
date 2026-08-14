import random
import backpropagation as bp
y_true = [0]*10
grad_L2_weights = [[0]*128]*10

d = bp.test_data  #10 000
ds =bp.dataset   #len 60 000

#print(   grad_L2_weights )
size = 10
indices = []
for i in range(size):
    indices.append(int(random.random()*60000))

print(indices)
