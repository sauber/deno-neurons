import { assertEquals, assertInstanceOf } from "@std/assert";
import { Network } from "./network.ts";
import { Scatter } from "./scatter.ts";
import type { Inputs, Outputs } from "./train.ts";
import { Train } from "./train.ts";

const xor = new Network(2).dense(3).lrelu.dense(1).sigmoid;

// XOR training set
const xs: Inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const ys: Outputs = [[0], [1], [1], [0]];

// Initialize object

Deno.test("Initialize", () => {
  const s = new Scatter(xor, xs, ys);
  // console.log(s);
  assertInstanceOf(s, Scatter);
});

Deno.test("Untrained and trained plot", () => {
  const s = new Scatter(xor, xs, ys);
  const initial = s.plot();
  console.log(initial);

  const train = new Train(xor, xs, ys);
  train.epsilon = 0.001;
  train.run(20000, 0.9);

  const trained = s.plot();
  console.log(trained);
});

Deno.test("Circle Training", () => {
  // Create network
  // const circle = new Network(2).dense(9).lrelu.dense(11).lrelu.dense(7).lrelu.dense(5).sigmoid;
  const circle = new Network(2).dense(8).lrelu.dense(6).lrelu.dense(4).lrelu.dense(1).tanh;
  // const circle = new Network(2).dense(7).lrelu.dense(5).lrelu;

  // Generate test data for a fat circle
  const xs: Inputs = [];
  const ys: Outputs = [];
  for (let i = 0; i<150; ++i) {
    const x = Math.random()*2-1;
    const y = Math.random()*2-1;
    const r = Math.sqrt(x*x+y*y);
    // Circle
    const c = (r >=0.45 && r<= 0.75) ? 1 : -1;
    // Wave
    const w = r < 0.5 ? (2*r)-0.5 : 1.5-(2*r);
    xs.push([x,y]);
    ys.push([c]);
  }
  // console.log({xs, ys})

  const s = new Scatter(circle, xs, ys);
  const initial = s.plot();
  console.log(initial);

  const train = new Train(circle, xs, ys);
  train.epsilon = 0.001;
  train.run(20000, 0.4);
  
  const trained = s.plot();
  console.log(trained);

})
