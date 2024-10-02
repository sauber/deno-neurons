import { NeuralNetwork } from "brainjs";
import { Dashboard } from "../src/dashboard.ts";

// Wave training data
const wave = [];
for (let i = 0; i < 150; ++i) {
  const x = Math.random() * 2 - 1;
  const y = Math.random() * 2 - 1;
  const r = Math.sqrt(x * x + y * y);
  const s = -Math.sin(r * 7);
  wave.push({ input: [x, y], output: [s] });
}

const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

// provide optional config object (or undefined). Defaults shown.
const config = {
  // binaryThresh: 0.5, // ¯\_(ツ)_/¯
  hiddenLayers: [3], // array of ints for the sizes of the hidden layers in the network
  activation: "tanh", // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh']
};

// create a simple feed forward neural network with backpropagation
const net = new NeuralNetwork(config);




const result = net.train(xor);
console.log(result);

const output = net.run([0, 0]); // [0.987]
console.log(output);
