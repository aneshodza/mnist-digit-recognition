import { NeuralNetwork, ActivationFunctions, OutputFunctions } from './network.js';
import { setupCanvas, getCenteredData } from './ui.js';

let nn = null;
const canvas = document.getElementById("canvas");
const resultEl = document.getElementById("result");

const ctx = setupCanvas("canvas", predict);

window.clearCanvas = () => {
  ctx.clearRect(0, 0, 28, 28);
  resultEl.innerText = `Prediction: ?`;
};

async function predict() {
  if (!nn) return;
  const input = getCenteredData(ctx, canvas);
  const output = nn.forward(input);
  const digit = output.indexOf(Math.max(...output));
  const confidence = (Math.max(...output) * 100).toFixed(2);
  resultEl.innerText = `Prediction: ${digit} (${confidence}%)`;
}

async function init() {
  try {
    const response = await fetch("model.json");
    const json = await response.text();
    nn = new NeuralNetwork(json, { ActivationFunctions, OutputFunctions });
  } catch (e) {
    console.error("Failed to load model", e);
  }
}

init();
