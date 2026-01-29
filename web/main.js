import { NeuralNetwork, ActivationFunctions, OutputFunctions } from './network.js';
import { setupCanvas, getCenteredData, updateInstructions, initBars, updateBars } from './ui.js';

let nn = null;
const canvas = document.getElementById("canvas");
const resultEl = document.getElementById("result");

const ctx = setupCanvas("canvas", predict);

window.clearCanvas = () => {
  ctx.clearRect(0, 0, 28, 28);
  updateBars(new Array(10).fill(0));
  resultEl.innerText = `Prediction: ?`;
};

async function predict() {
  if (!nn) return;
  const input = getCenteredData(ctx, canvas);
  const output = nn.forward(input);
  updateBars(output);
  const digit = output.indexOf(Math.max(...output));
  const confidence = (Math.max(...output) * 100).toFixed(2);
  resultEl.innerText = `Prediction: ${digit} (${confidence}%)`;
}

async function init() {
  updateInstructions();
  initBars();
  try {
    const response = await fetch("model.json");
    const json = await response.text();
    nn = new NeuralNetwork(json, { ActivationFunctions, OutputFunctions });
  } catch (e) {
    console.error("Failed to load model", e);
  }
}

init();
