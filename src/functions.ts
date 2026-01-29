import type { ActivationFunction, OutputFunction, Scalar, Vector } from "./types.js";

export const ActivationFunctions = {
  relu: {
    forward: (z: Vector): Vector => z.map((x) => Math.max(0, x)),
    backward: (z: Vector): Vector => z.map((x) => (x > 0 ? 1 : 0)),
  },
  identity: {
    forward: (z: Vector): Vector => z,
    backward: (z: Vector): Vector => new Float32Array(z.length).fill(1),
  }
} satisfies Record<string, ActivationFunction>;

export const OutputFunctions = {
  softmax: {
    output: (z: Vector): Vector => {
      const maxZ = Math.max(...z);
      const exps = z.map((x) => Math.exp(x - maxZ));
      const sum = exps.reduce((a, b) => a + b);
      return exps.map((x) => x / sum);
    },
    cost: (y_pred: Vector, y_target: Vector): Scalar => {
      return -y_target.reduce((acc, y, i) => acc + y * Math.log(y_pred[i]! + 1e-10), 0);
    },
    error: (y_pred: Vector, y_target: Vector): Vector => {
      return y_pred.map((y, i) => y - y_target[i]!);
    }
  },
} satisfies Record<string, OutputFunction>;
