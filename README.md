# MNIST Digit Recognition

A lightweight, library-free implementation of a neural network in vanilla JavaScript and TypeScript designed to recognize handwritten digits from the **MNIST dataset**. This project includes a training engine that runs in Node.js and a real-time web interface where users can draw digits and get instant predictions.

## 🚀 Features
* **Built from Scratch**: Core neural network logic (forward and backward propagation) implemented without external machine learning libraries like TensorFlow or PyTorch.
* **Node.js Training**: Train your model on the original MNIST binary dataset (`.idx` files) directly in your terminal.
* **Web Interface**: A responsive HTML5 canvas for drawing digits with real-time "MNIST-style" preprocessing (centering by center of mass).
* **Modular Code**: Clearly separated logic for neural network math, training utilities, and UI components.

## 🛠 Tech Stack
* **Language**: TypeScript (Source), JavaScript (Web)
* **Environment**: Node.js (Training), Modern Web Browser (Inference)
* **Data**: MNIST Database of handwritten digits (60,000 training samples, 10,000 test samples).

## 📂 Project Structure
* `src/`: TypeScript source files for the neural network and training logic.
    * `neural_network.ts`: Core `NeuralNetwork` and `Layer` classes.
    * `functions.ts`: Activation (ReLU, Identity) and Output (Softmax) functions.
    * `main.ts`: Entry point for training and evaluation.
* `web/`: Files for the web-based digit recognizer.
    * `index.html`: The user interface.
    * `ui.js`: Handles canvas drawing and digit centering.
    * `network.js`: Browser-compatible version of the neural network engine.
* `archive/`: Storage for the raw MNIST binary dataset (`idx3-ubyte` format).

## ⚙️ How to Use

### 1. Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/aneshodza/mnist-digit-recognition.git
cd mnist-digit-recognition
npm install
```

### 2. Training the Model

To train the neural network on the MNIST dataset:
1. Ensure the binary data files are in the `archive/` directory.
2. Run the training & evaluation script:
```bash
npm run start
```
3. Once training is complete, the script will export the weights to a model.json file.

### 3. Running the Web Interface

To use the trained model in your browser:
1. Move the generated `model.json` into the `web/` folder.
2. Open `web/index.html`.
3. Draw a digit on the 28x28 canvas and see the prediction results.

## 📊 Model Details

The system typically uses a multilayer perceptron (MLP) architecture:
* **Input Layer:** 784 neurons (representing 28x28 pixel values).
* **Hidden Layers:** Configurable layers with ReLU activation.
* **Output Layer:** 10 neurons with Softmax activation (one for each digit 0-9).

When training on 50'000 images and testing on 9'000 it achieved a precision of around
95%, while the Web-UI likely achieves a lower precision than that.
