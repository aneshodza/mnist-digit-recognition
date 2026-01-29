import fs from "fs";
import type { TrainingPair, Vector } from "./types.js";

const rawImages: Buffer = fs.readFileSync("archive/train-images.idx3-ubyte");
const imageData: Uint8Array = new Uint8Array(rawImages.buffer, 16);

const rawLabels: Buffer = fs.readFileSync("archive/train-labels.idx1-ubyte");
const labelData: Uint8Array = new Uint8Array(rawLabels.buffer, 8);

function getNormalizedImage(index: number): Float32Array {
  const start = index * 784;
  const pixels = imageData.slice(start, start + 784);
  return Float32Array.from(pixels, (p: number) => p / 255);
}

function getTrainingPair(index: number): TrainingPair {
  const x = getNormalizedImage(index);

  const y_label = labelData[index] ?? 0;
  const y_target = new Float32Array(10);
  y_target[y_label] = 1;

  return { x, y_target, label: y_label };
}

function* trainingDataGenerator(): Generator<TrainingPair> {
  const numSamples = labelData.length;
  for (let i = 0; i < numSamples; i++) {
    yield getTrainingPair(i);
  }
}

function visualizeImage(image: Vector): void {
  for (let i = 0; i < 28; i++) {
    let row = "";
    for (let j = 0; j < 28; j++) {
      const pixelValue = image[i * 28 + j]!;
      row += pixelValue > 0.5 ? "##" : "..";
    }
    console.log(row);
  }
}

export { trainingDataGenerator, visualizeImage };
