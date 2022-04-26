<h1 align="center">Handwriting-recognition</h1>

- Handwriting Recognition Using NUMPY 
- This is an exercise using multiple neurons. Assuming that we only have 0, 1, and 2 to identify, then there must be 3 neural network outputs. If it is 0, output [1,0,0], and if it is 1, output [0,1,0], if it is 2, output [0,0,1].
- Through the basic theory of neural network, we can know that adding Hidden layer can help the learning process. This exercise only adds one layer to try.
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_1.png" width="80%"></p>
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_4.png" width="40%"></p>

- This formula is the main training content to be performed in this example. The **Activation Function** inside represents the input layer to the hidden layer, and the whole picture of the formula represents the hidden layer to the output layer.

## Prepare the dataset
- This example uses the famous MNIST data set, go to the website to download four files. 
http://yann.lecun.com/exdb/mnist/

<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_2.png" width="80%"></p>


- Input layer: Each material in the data is a combination of 28*28, which is first converted into a sample of 784 features, and then learned by the neural network.

- Hidden Layer: This exercise uses a hidden layer with a number of 100 neurons.

- Output layer: Because the numbers 0-9 are to be recognized, there are 10 outputs in total.

- In the dataset, there are a total of 60,000 handwritten pictures to train neurons, and this structure has a total of 79,510 weight parameters.

<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_3.png" width="60%"></p>

- **Step1**: We use 70,000 handwritten pictures for training and 10,000 for testing.
- **Step2**: Train the neural network
- **Step3**: Test results

```
import numpy as np
import load_mnist as lm
dataset = lm.load_mnist()

print(dataset)
```
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_5.png" width="60%"></p>

-  The load_mnist function will convert the downloaded 4 files into 4 arrays and store them in the dictionary.

## Start training the neural network

- In the early stage of training, we randomly assign all the w and b weight values, and then feed them to the neural network, using the concept of loss function to minimize the error between the predicted value and the true value.
- In this example, the **Loss function** is calculated by the **MSE** method.
- In order to find the optimal weight configuration that "minimizes the value of the loss function", the loss function L performs partial differentiation on all weightsIn order to find the optimal weight configuration that "minimizes the value of the loss function", the loss function L performs partial differentiation on all weights, and the obtained values continue to use the update formula to calculate new weights, thus completing a training session.
- But because this exercise has 79,510 weights, in order to increase efficiency, **back propagation (backprop, BP)** will be used.


## Weight initialization function
 **neuralnet.py**
```
import numpy as np

# 輸入每層神經元數目的陣列，例如 shape_list = [784, 100, 10]
def make_params(shape_list):
    w_list = []
    b_list = []
    for i in range(len(shape_list)-1):
        
        # 產生初始值為遵從標準常態分佈的亂數
        weight = np.random.randn(shape_list[i], shape_list[i+1])
        
       # 始值全部設定 0.1
        bias = np.ones(shape_list[i+1])/10.0
    
        w_list.append(weight)
        b_list.append(bias)
    
    return w_list, b_list
    #傳回的結果：w_list與b_list都是2個元素的list
```

```
def sigmoid(x): # sigmoid 函式
    return 1/(1+np.exp(-x))
```
- This is the **Activation Function**
```
def inner_product(x_train, w, b):# 內積再加上偏權值
    return np.dot(x_train, w)+ b
```
- Inner product plus bias
```
def activation(x_train, w, b):
    return sigmoid(inner_product(x_train, w, b))
```
- Inside represents the input layer to the hidden layer, and the whole picture of the formula represents the hidden layer to the output layer.

```
def calculate(x_train, w_list, b_list):
    
    val_dict = {}

    a_1 = inner_product(x_train, w_list[0], b_list[0]) # (N, 100)
    y_1 = sigmoid(a_1) # (N, 100)
    a_2 = inner_product(y_1, w_list[1], b_list[1]) # (N, 10)
    y_2 = sigmoid(a_2)
    y_2 /= np.sum(y_2, axis=1, keepdims=True)  # 這裡進行簡單的正規化
    val_dict['y_1'] = y_1
    val_dict['y_2'] = y_2
    #算好後存回字典
    
    return val_dict
    
```
```
def update(x_train, w_list, b_list, y_train, eta):
  
    val_dict = {}
    val_dict = calculate(x_train, w_list, b_list)
    y_1 = val_dict['y_1']
    y_2 = val_dict['y_2']

    #取用calculate()函式算出來的y1及y2
    
    d12_d11 = 1.0
    d11_d9 = 1/x_train.shape[0]*(y_2 - y_train)
    d9_d8 = y_2*(1.0 - y_2)
    d8_d7 = 1.0
    d8_d6 = np.transpose(y_1)
    d8_d5 = np.transpose(w_list[1])
    d5_d4 = y_1 * (1 - y_1)
    d4_d3 = 1.0
    d4_d2 = np.transpose(x_train)

    #一一計算各個局部偏微分

    #以下計算 d12_d7_&_d12_d6
    d12_d8 = d12_d11 * d11_d9 * d9_d8

    b_list[1] -= eta*np.sum(d12_d8 * d8_d7, axis=0)
    w_list[1] -= eta*np.dot(d8_d6, d12_d8)
    #作為權重更新，eta為學習率

    #以下計算 d12_d3_&_d12_d2
    d12_d8 = d12_d11 * d11_d9 * d9_d8
    d12_d5 = np.dot(d12_d8, d8_d5)
    d12_d4 = d12_d5 * d5_d4
        
    b_list[0] -= eta * np.sum(d12_d4 * d4_d3, axis=0)
    w_list[0] -= eta * np.dot(d4_d2, d12_d4)

    return w_list, b_list

```

## Start training the neural network
 **learn.py**
 
 ```
import numpy as np
import neuralnet as nl
import load_mnist as lm

dataset = lm.load_mnist()
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']


w_list, b_list = nl.make_params([784, 100, 10])


for epoch in range(1):
#先訓練一次
    ra = np.random.randint(60000,size=60000)
    for i in range(60):
        x_batch = x_train[ra[i*1000:(i+1)*1000],:]
        y_batch = y_train[ra[i*1000:(i+1)*1000],:]
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, eta=2.0)

```
 
```
y_test[0:10]
```
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_6.png" width="60%"></p>

- Here are the correct answers for the first 10 groups.


## Verify performance

```
val_dict = nl.calculate(x_test, w_list, b_list)
print(val_dict['y_2'][0:10].round(2))
```

- Pass in x_test and the weight result calculated earlier.
- Take the training result from the dictionary and display the first 10 groups.

<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_7.png" width="60%"></p>

- But the display results are not very accurate, and then try to train 300 times.

```
for epoch in range(300):
#改為訓練300次
    ra = np.random.randint(60000,size=60000)
    for i in range(60):
        x_batch = x_train[ra[i*1000:(i+1)*1000],:]
        y_batch = y_train[ra[i*1000:(i+1)*1000],:]
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, eta=2.0)

```
<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_8.png" width="60%"></p>

- It is found that group 8 is still incorrect, but the value is very high.
- Therefore, use matplotlib to call this group of pictures to check.

```
import matplotlib.pyplot as plt
plt.imshow(dataset['x_test'][8].reshape(28,28),cmap="gray")
plt.axis("off")
plt.show()
```

<p align="center"><img src="https://github.com/chiardy90/readme_pic/blob/main/Handwriting-recognition/Handwriting-recognition_9.png" width="15%"></p>

- It can be found that this picture is actually very difficult to identify. No matter how good the method is, it is difficult to deal with this kind of garbage, so data cleaning is very important.
