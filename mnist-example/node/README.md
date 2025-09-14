# Image Recognition Application

An application for image recognition using neural networks based on the `neuro-lib` library.

## Features

- **Neural Network Training** on image datasets
- **Image Recognition** with detailed statistics
- **Modular Architecture** with separation of concerns
- **Support for Various Activation Functions** (ReLU, Sigmoid, Softmax)
- **Automatic Weight Initialization** (He, Xavier)
- **Model Saving and Loading** in JSON format

## Architecture

The application is divided into modules:

### 📁 Module Structure

- **`src/index.ts`** - Main application file
- **`src/cli.ts`** - Command line argument processing
- **`src/imageProcessor.ts`** - Image loading and processing
- **`src/trainer.ts`** - Neural network training
- **`src/recognizer.ts`** - Image recognition
- **`src/utils.ts`** - Helper functions for file operations

### 🧠 Neural Network

- **Hidden Layers**: ReLU activation
- **Output Layer**: Softmax activation
- **Loss Function**: CrossEntropy
- **Weight Initialization**: He for ReLU, Xavier for Sigmoid

## Usage

### Install Dependencies

```bash
yarn install
```

### Train Model

```bash
# Create new model (automatic activation function selection)
yarn start -t -m model.json -f ./images -l 784,512,256,96,10 -e 100 -s 0.001

# Create model with custom activation functions
yarn start -t -m model.json -f ./images -l 784,128,10 -a ReLU,LeakyReLU,Softmax -e 100 -s 0.001

# Continue training existing model
yarn start -t -m model.json -f ./images -e 50 -s 0.0005
```

### Recognize Images

```bash
yarn start -r -m model.json -f ./test_images
```

### Help

```bash
yarn start --help
```

## Command Line Parameters

### Main Options

- `-t, --train` - Training mode
- `-r, --recognize` - Recognition mode
- `-m, --model <fileName>` - Model filename
- `-h, --help` - Show help

### Training Options

- `-f, --folder <folderName>` - Folder with images for training
- `-l, --layers <layers>` - Layer configuration (e.g.: 784,512,256,96,10)
- `-a, --activations <activations>` - Activation functions for each layer (ReLU,LeakyReLU,Sigmoid,Softmax)
- `-e, --epochs <number>` - Number of training epochs (default: 100)
- `-s, --speed <number>` - Learning rate (default: 0.001)

### Recognition Options

- `-f, --folder <folderName>` - Folder with images for recognition

## Data Format

### Image Folder Structure

```
images/
├── 0/          # Class 0
│   ├── img1.png
│   └── img2.png
├── 1/          # Class 1
│   ├── img1.png
│   └── img2.png
└── ...
```

### Image Format

- **Format**: PNG
- **Normalization**: Pixels normalized to range [0, 1]
- **Channels**: Only red channel used

### Activation Functions

- **ReLU**: Rectified Linear Unit - standard for hidden layers
- **LeakyReLU**: Leaky ReLU - improved version of ReLU
- **Sigmoid**: Sigmoid function - for hidden layers
- **Softmax**: For output layer in classification tasks

### Automatic Loss Function Selection

- **Softmax in last layer** → automatically selects **CrossEntropy**
- **Any other function** → automatically selects **MSE**
