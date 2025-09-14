# Neural Network Library in TypeScript

[:rocket: LIVE DEMO](https://digition.netlify.app) of digit recognition using React.

## Overview

This project provides a comprehensive neural network library built in TypeScript with practical applications for image recognition. The library includes:

- **Core Library** (`lib/`) - Neural network implementation with various activation and loss functions
- **Node.js Application** (`example/node/`) - Command-line tool for training and recognition
- **React Application** (`example/react/`) - Interactive web interface for digit recognition

## Features

### Neural Network Library

- Multiple activation functions (ReLU, LeakyReLU, Sigmoid, Softmax)
- Loss functions (MSE, CrossEntropy)
- Automatic loss function selection based on output layer
- JSON serialization/deserialization
- Comprehensive test coverage

### Node.js Application

- Train neural networks on image datasets
- Recognize images with detailed statistics
- Support for custom activation functions per layer
- Automatic model saving and loading

### React Application

- Interactive drawing interface
- Real-time digit recognition
- Probability visualization
- Modern, responsive UI

## Quick Start

### Library Usage

```typescript
import { Network, LayerConfig } from "./lib/src/network.ts";
import { ActivationFunctionCollection } from "./lib/src/functions/activation.ts";

// Create network configuration
const layerConfigs: LayerConfig[] = [
	{ neurons: 784, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 128, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 10, activationFunction: ActivationFunctionCollection.Softmax },
];

// Create network (loss function auto-selected)
const network = new Network(layerConfigs);

// Train the network
network.setInputSignals(inputData).forward().backward(expectedOutput, 0.01);
```

### Node.js Application

```bash
# Install dependencies
yarn install

# Train a new model
yarn start -t -m model.json -f ./images -l 784,128,10 -e 100 -s 0.001

# Recognize images
yarn start -r -m model.json -f ./test_images
```

### React Application

```bash
cd example/react
yarn install
yarn dev
```

## Architecture

The library follows a modular architecture:

- **Network** - Main neural network class
- **Layer** - Individual network layers with activation functions
- **Neuron** - Individual neurons with weights and biases
- **Activation Functions** - ReLU, LeakyReLU, Sigmoid, Softmax
- **Loss Functions** - MSE, CrossEntropy with automatic selection
