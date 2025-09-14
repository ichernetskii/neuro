# neuro-lib

A library for creating and training neural networks with support for various activation functions and loss functions.

## Features

### Activation Functions

- **ReLU** - Rectified Linear Unit
- **LeakyReLU** - Leaky Rectified Linear Unit
- **Sigmoid** - Sigmoid function
- **Softmax** - Softmax function (applied only to output layer)

### Loss Functions

- **MSE** (Mean Squared Error) - Mean squared error
- **CrossEntropy** - Cross entropy (optimal for classification with Softmax)

All loss functions are implemented as objects with methods for computing losses and their derivatives.

## Usage

### Creating a Network with Layer Configuration

```typescript
import { Network, LayerConfig } from "./src/network.ts";
import { ActivationFunctionCollection } from "./src/functions/activation.ts";
import { LossFunctionCollection } from "./src/functions/loss.ts";

// Create layer configuration (without loss function)
const layerConfigs: LayerConfig[] = [
	{ neurons: 4, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 8, activationFunction: ActivationFunctionCollection.LeakyReLU },
	{ neurons: 6, activationFunction: ActivationFunctionCollection.Sigmoid },
	{ neurons: 3, activationFunction: ActivationFunctionCollection.Softmax },
];

// Create network (loss function selected automatically)
const network = new Network(layerConfigs);

// Set input data
network.setInputSignals([0.5, 0.3, 0.8, 0.2]);

// Forward pass
network.forward();

// Get outputs (class probabilities)
const outputs = network.getOutputSignals().map(signal => signal.value);

// Train with Cross Entropy
const expectedOutput = [0, 1, 0]; // One-hot encoding for class 1
network.backward(expectedOutput, 0.1);

// Calculate loss
const loss = network.calculateLoss(expectedOutput);
```

### Architecture Examples

```typescript
// Regression network
const regressionConfigs: LayerConfig[] = [
	{ neurons: 3, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 10, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 5, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 1, activationFunction: ActivationFunctionCollection.ReLU },
];
const regressionNetwork = new Network(regressionConfigs); // MSE automatically

// Classification network
const classificationConfigs: LayerConfig[] = [
	{ neurons: 5, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 8, activationFunction: ActivationFunctionCollection.Sigmoid },
	{ neurons: 3, activationFunction: ActivationFunctionCollection.Softmax },
];
const classificationNetwork = new Network(classificationConfigs); // CrossEntropy automatically

// Deep network with LeakyReLU
const deepConfigs: LayerConfig[] = [
	{ neurons: 10, activationFunction: ActivationFunctionCollection.LeakyReLU },
	{ neurons: 20, activationFunction: ActivationFunctionCollection.LeakyReLU },
	{ neurons: 15, activationFunction: ActivationFunctionCollection.LeakyReLU },
	{ neurons: 10, activationFunction: ActivationFunctionCollection.LeakyReLU },
	{ neurons: 5, activationFunction: ActivationFunctionCollection.LeakyReLU },
];
const deepNetwork = new Network(deepConfigs); // MSE automatically

// Mixed architecture
const mixedConfigs: LayerConfig[] = [
	{ neurons: 4, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 8, activationFunction: ActivationFunctionCollection.LeakyReLU },
	{ neurons: 6, activationFunction: ActivationFunctionCollection.Sigmoid },
	{ neurons: 3, activationFunction: ActivationFunctionCollection.Softmax },
];
const mixedNetwork = new Network(mixedConfigs); // CrossEntropy automatically

// Network with MSE by default
const defaultConfigs: LayerConfig[] = [
	{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 3, activationFunction: ActivationFunctionCollection.ReLU },
];
const defaultNetwork = new Network(defaultConfigs); // MSE automatically
```

### Features

#### Layer Configuration

- Each layer is configured through a `LayerConfig` object
- Required parameters: `neurons` (number of neurons) and `activationFunction`
- Loss function is specified separately in the `Network` constructor
- Automatic weight initialization depending on the layer's activation function

#### Loss Functions

- Loss functions are implemented as objects with `loss()` and `derivative()` methods
- Supports MSE and CrossEntropy
- **Automatic selection**: Softmax in last layer → CrossEntropy, otherwise → MSE
- Loss function can be specified explicitly as second parameter in `Network` constructor

#### Softmax

- Softmax is applied only to the output layer
- Softmax outputs always sum to 1 (class probabilities)
- CrossEntropy works optimally with Softmax for classification tasks

### Saving and Loading

```typescript
// Save
const json = network.toJSON();

// Load (automatically restores activation and loss functions)
const loadedNetwork = Network.fromJSON(json);
```

### Error Handling

```typescript
// Error with empty layer configuration
try {
    const network = new Network([]);
} catch (error) {
    console.log(error.message); // "Network must contain at least one layer"
}

// Error with unknown loss function during deserialization
try {
    const network = Network.fromJSON({ layers: [{ neurons: [...], activationFunction: "ReLU", lossFunction: "UnknownLoss" }] });
} catch (error) {
    console.log(error.message); // "Unknown loss function: UnknownLoss"
}
```

### Creating Custom Loss Functions

```typescript
import { LossFunction } from "./src/functions/loss.ts";

const CustomLoss: LossFunction = (predicted: number[], expected: number[]): number => {
	// Your loss function implementation
	return predicted.reduce((sum, pred, index) => sum + Math.abs(pred - expected[index]), 0);
};

CustomLoss.derivative = (predicted: number[], expected: number[]): number[] => {
	// Derivative of your loss function
	return predicted.map((pred, index) => (pred > expected[index] ? 1 : -1));
};

CustomLoss.functionName = "CustomLoss";

// Usage
const customConfigs: LayerConfig[] = [
	{ neurons: 3, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 5, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
];
const network = new Network(customConfigs, CustomLoss); // Explicit loss function specification
```

## Module Structure

The library is organized in a modular structure:

### Core Modules

- **`src/network.ts`** - Main Network class and interfaces
- **`src/layer.ts`** - Layer class for neural network layers
- **`src/neuron.ts`** - Neuron class and helper classes

### Function Modules

- **`src/functions/activation.ts`** - Activation functions (ReLU, LeakyReLU, Sigmoid, Softmax)
- **`src/functions/loss.ts`** - Loss functions (MSE, CrossEntropy)
- **`src/functions/random.ts`** - Functions for random number generation and weight initialization
- **`src/functions/round.ts`** - Helper functions for rounding

### Tests

- **`src/test/`** - Complete test coverage for all modules (59 tests)

### Imports

```typescript
// Core classes
import { Network, LayerConfig } from "./src/network.ts";

// Activation functions
import { ActivationFunctionCollection } from "./src/functions/activation.ts";

// Loss functions
import { LossFunctionCollection } from "./src/functions/loss.ts";

// Helper functions
import { randomNormalHe, randomUniformXavier } from "./src/functions/random.ts";
import { round } from "./src/functions/round.ts";
```

### Typed Union Types

The library uses union types for type-safe function names:

```typescript
// Activation functions
type ActivationFunctionName = "ReLU" | "LeakyReLU" | "Sigmoid" | "Softmax";

// Loss functions
type LossFunctionName = "MSE" | "CrossEntropy";

// Usage
ActivationFunctionCollection.get("ReLU"); // ✅ Type-safe
ActivationFunctionCollection.get("InvalidFunc"); // ❌ TypeScript error

// New convenient API
ActivationFunctionCollection.ReLU.derivative([1, 2, 3]); // ✅ Direct access
ActivationFunctionCollection.all(); // ✅ All functions
ActivationFunctionCollection.names(); // ✅ All names
```

### Classes with Static Methods

The library uses objects with static methods for convenient function access:

```typescript
// Activation functions
ActivationFunctionCollection.ReLU(values); // Direct call
ActivationFunctionCollection.ReLU.derivative(values); // Derivative
ActivationFunctionCollection.get("ReLU"); // Get by name
ActivationFunctionCollection.all(); // All functions
ActivationFunctionCollection.names(); // All names

// Loss functions
LossFunctionCollection.MSE(predicted, expected); // Direct call
LossFunctionCollection.MSE.derivative(pred, exp); // Derivative
LossFunctionCollection.get("MSE"); // Get by name
LossFunctionCollection.all(); // All functions
LossFunctionCollection.names(); // All names
```

### Examples of New API Usage

```typescript
import { Network, LayerConfig } from "./src/network.ts";
import { ActivationFunctionCollection } from "./src/functions/activation.ts";
import { LossFunctionCollection } from "./src/functions/loss.ts";

// Create network with new API
const layerConfigs: LayerConfig[] = [
	{ neurons: 4, activationFunction: ActivationFunctionCollection.ReLU },
	{ neurons: 8, activationFunction: ActivationFunctionCollection.LeakyReLU },
	{ neurons: 6, activationFunction: ActivationFunctionCollection.Sigmoid },
	{ neurons: 3, activationFunction: ActivationFunctionCollection.Softmax },
];

const network = new Network(layerConfigs); // CrossEntropy automatically

// Direct access to functions
const reluOutput = ActivationFunctionCollection.ReLU([-1, 0, 1]); // [0, 0, 1]
const reluDerivative = ActivationFunctionCollection.ReLU.derivative([-1, 0, 1]); // [0, 1, 1]

// Get function by name
const sigmoidFunc = ActivationFunctionCollection.get("Sigmoid");

// All available functions
const allActivations = ActivationFunctionCollection.all();
const activationNames = ActivationFunctionCollection.names();
```

This provides simplicity, type safety, ease of use, and excellent Node.js compatibility.
