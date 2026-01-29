import { ActivationFunctions, OutputFunctions } from "./functions.js";
import { NeuralNetwork } from "./neural_network.js";
import { trainingDataGenerator } from "./training_utils.js";

const generator = trainingDataGenerator();

const neuralNetwork = new NeuralNetwork(OutputFunctions.softmax);
neuralNetwork.addHiddenLayer(ActivationFunctions.relu, 784, 128);
neuralNetwork.addHiddenLayer(ActivationFunctions.identity, 128, 10);

const iterations = 50000
let percentage = 0
const visitor = (i: number) => {
  const newPrecentage = Math.round(i/iterations * 100)
  if (newPrecentage !== percentage) {
    percentage = newPrecentage
    console.log(`${percentage}% done with training`)
  }
}
neuralNetwork.train(generator, iterations, 0.01, visitor)
console.log('Model is done with training. Evaluating precision now...')
const precision = neuralNetwork.evaluate(generator, 9000)
console.log(`We achieved a precision of ${Math.round(precision * 100)}%`);
neuralNetwork.export()
