import type { Layer } from "./neural_network.js";

type Scalar = number;
type Vector = Float32Array;
type Matrix = Float32Array[];

type ForwardData = { signal: Vector, z: Vector, layer: Layer };

interface ActivationFunction {
  forward(z: Vector): Vector;
  backward(z: Vector): Vector;
}
interface OutputFunction {
  output(z: Vector): Vector;
  cost(y_pred: Vector, y_target: Vector): number,
  error(y_pred: Vector, y_target: Vector): Vector;
} 

interface TrainingPair {
  x: Float32Array;
  y_target: Float32Array;
  label: number;
}

export type { Scalar, Vector, Matrix, ForwardData, ActivationFunction, OutputFunction, TrainingPair }
