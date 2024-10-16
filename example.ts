import { Network, Train } from "jsr:@sauber/neurons";

// Generate training inputs and outputs
const inputs: number[][] = [];
const outputs: number[][] = [];
for (let i = 0; i < 100; ++i) {
  const [x, y] = [Math.random() * 14 - 7, Math.random() * 14 - 7];
  inputs.push([x, y]);
  outputs.push([-Math.sin(Math.sqrt(x * x + y * y))]);
}

// Define neural network
const network = new Network(2).dense(8).lrelu.dense(6).lrelu.dense(1).tanh;

// Perform training of neural network
const train = new Train(network, inputs, outputs);
train.epsilon = 0.01;
train.callback = (iteration: number, loss: number[]) =>
  console.log(`Iteration: ${iteration}, Loss: ${loss[loss.length - 1]}`);
const max_iteration = 20000;
const learning_rate = 0.2;
train.run(max_iteration, learning_rate);
