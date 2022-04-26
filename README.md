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

## Train

- **Step1**: We use 70,000 handwritten pictures for training and 10,000 for testing.
- **Step2**: Train the neural network
- **Step3**: Test results
