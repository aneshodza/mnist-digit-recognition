import fs from "fs";
import type {
  ActivationFunction,
  ForwardData,
  Matrix,
  OutputFunction,
  TrainingPair,
  Vector,
} from "./types.js";
import { ActivationFunctions, OutputFunctions } from "./functions.js";

export class Layer {
  weights: Matrix;
  biases: Vector;
  activationFunction: ActivationFunction;

  constructor(
    weights: Matrix,
    biases: Vector,
    activationFunction: ActivationFunction,
  ) {
    this.weights = weights;
    this.biases = biases;
    this.activationFunction = activationFunction;
  }
}

class NeuralNetwork {
  layers: Layer[];
  outputFunction: OutputFunction;

  constructor(outputFunction: OutputFunction) {
    this.layers = [];
    this.outputFunction = outputFunction;
  }

  addHiddenLayer(
    activationFunction: ActivationFunction,
    inputSize: number,
    outputSize: number,
  ): void {
    const weights: Matrix = [];
    for (let i = 0; i < outputSize; i++) {
      const row: Vector = new Float32Array(inputSize);
      for (let j = 0; j < inputSize; j++) {
        row[j] = Math.random() * 0.01;
      }
      weights.push(row);
    }
    const biases: Vector = new Float32Array(outputSize);

    const layer = new Layer(weights, biases, activationFunction);
    this.layers.push(layer);
  }

  forward(
    x: Vector,
    fn: ((signal: Vector, z: Vector, layer: Layer) => void) | null = null,
  ): Vector {
    let signal = x;
    for (const layer of this.layers) {
      const weights = layer.weights;

      const z = Float32Array.from(
        weights.map((neuron, i) => {
          const dotProduct = neuron
            .map((weight, j) => weight * signal[j]!)
            .reduce((a, b) => a + b);

          return dotProduct + layer.biases[i]!;
        }),
      );
      if (fn) fn(signal, z, layer);
      signal = layer.activationFunction.forward(z);
    }

    return this.outputFunction.output(signal);
  }

  backwards(deltas: Vector, forwardData: ForwardData[], alpha: number) {
    for (const data of forwardData) {
      const { signal, z, layer } = data;

      const localDeltas: Vector = new Float32Array(deltas.length);
      for (let i = 0; i < deltas.length; i++) {
        localDeltas[i] = deltas[i]! * layer.activationFunction.backward(z)[i]!;
      }

      const nextDeltas: Vector = new Float32Array(signal.length);
      for (let j = 0; j < signal.length; j++) {
        let errSum = 0;
        for (let i = 0; i < layer.weights.length; i++) {
          errSum += localDeltas[i]! * layer.weights[i]![j]!;
        }
        nextDeltas[j] = errSum;
      }

      for (let i = 0; i < layer.biases.length; i++) {
        layer.biases[i] = layer.biases[i]! - alpha * localDeltas[i]!;
      }

      for (let i = 0; i < layer.weights.length; i++) {
        const neuron: Vector = layer.weights[i]!;
        for (let j = 0; j < neuron.length; j++) {
          neuron[j]! -= alpha * localDeltas[i]! * signal[j]!;
        }
      }

      deltas = nextDeltas;
    }
  }

  train(
    generator: Generator<TrainingPair>,
    iterations = 1000,
    alpha = 0.01,
    fn: ((i: number) => void) | null = null,
  ) {
    let forwardData: ForwardData[] = [];
    const visitor = (signal: Vector, z: Vector, layer: Layer) => {
      forwardData.unshift({ signal, z, layer });
    };
    for (let i = 0; i < iterations; i++) {
      const { x, y_target } = generator.next().value;
      const y_pred = this.forward(x, visitor);
      const deltas = this.outputFunction.error(y_pred, y_target);
      this.backwards(deltas, forwardData, alpha);

      if (fn) fn(i);
      forwardData = [];
    }
  }

  evaluate(generator: Generator<TrainingPair>, iterations = 1000) {
    let correct = 0;
    for (let iteration = 0; iteration < iterations; iteration++) {
      const { x, label } = generator.next().value;
      const y_pred = this.forward(x);

      let max = 0;
      let pred = -1;
      for (let i = 0; i < y_pred.length; i++) {
        if (max < y_pred[i]!) {
          max = y_pred[i]!;
          pred = i;
        }
      }

      if (pred === label) correct++;
    }

    return correct / iterations;
  }

  export(filename = "model.json") {
    const data = {
      activationNames: this.layers.map((l) => {
        return Object.keys(ActivationFunctions).find(
          (key) =>
            ActivationFunctions[key as keyof typeof ActivationFunctions] ===
            l.activationFunction,
        );
      }),
      outputName: Object.keys(OutputFunctions).find(
        (key) =>
          OutputFunctions[key as keyof typeof OutputFunctions] ===
          this.outputFunction,
      ),
      layers: this.layers.map((layer) => ({
        weights: layer.weights.map((row) => Array.from(row)),
        biases: Array.from(layer.biases),
      })),
    };

    try {
      const json = JSON.stringify(data, null, 2);
      fs.writeFileSync(filename, json, "utf8");
      console.log(`Model successfully saved to ${filename}`);
    } catch (err) {
      console.error("Failed to save the model:", err);
    }
  }
}

export { NeuralNetwork };
