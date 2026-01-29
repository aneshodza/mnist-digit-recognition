export const ActivationFunctions = {
  relu: { forward: (z) => z.map((x) => Math.max(0, x)) },
  identity: { forward: (z) => z },
};

export const OutputFunctions = {
  softmax: {
    output: (z) => {
      const maxZ = Math.max(...z);
      const exps = z.map((x) => Math.exp(x - maxZ));
      const sum = exps.reduce((a, b) => a + b);
      return exps.map((x) => x / sum);
    },
  },
};

export class Layer {
  constructor(weights, biases, activationFunction) {
    this.weights = weights;
    this.biases = biases;
    this.activationFunction = activationFunction;
  }
}

export class NeuralNetwork {
  constructor(jsonString, config) {
    this.layers = [];
    const data = JSON.parse(jsonString);
    this.outputFunction = config.OutputFunctions[data.outputName];

    this.layers = data.layers.map((layerData, i) => {
      const weights = layerData.weights.map((row) => new Float32Array(row));
      const biases = new Float32Array(layerData.biases);
      const activation = config.ActivationFunctions[data.activationNames[i]];
      return new Layer(weights, biases, activation);
    });
  }

  forward(x) {
    let signal = x;
    for (const layer of this.layers) {
      const weights = layer.weights;
      const z = Float32Array.from(
        weights.map((neuron, i) => {
          let dot = 0;
          for (let j = 0; j < neuron.length; j++) dot += neuron[j] * signal[j];
          return dot + layer.biases[i];
        })
      );
      signal = layer.activationFunction.forward(z);
    }
    return this.outputFunction.output(signal);
  }
}
