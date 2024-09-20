import { assertInstanceOf, assertMatch } from "@std/assert";
import { Dashboard } from "./dashboard.ts";
import { Network } from "./network.ts";
import { Train } from "./train.ts";
import type { Inputs, Outputs } from "./train.ts";

// XOR training set
const xs: Inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const ys: Outputs = [[0], [1], [1], [0]];

const network: Network = new Network(2)
  .dense(5)
  .lrelu.dense(2)
  .lrelu.dense(1).tanh;

const train = new Train(network, xs, ys);
train.epsilon = 0.01;
// train.run(200000, 0.9);

Deno.test("Instance", () => {
  const d = new Dashboard(train);
  assertInstanceOf(d, Dashboard);
});

Deno.test("No iterations", () => {
  const d = new Dashboard(train);
  const v: string = d.render(0, []);
  assertMatch(v, /No data/);
  // console.log(v);
});

Deno.test("Iteration 1", () => {
  const d = new Dashboard(train);
  const v: string = d.render(1, [1]);
  assertMatch(v, /No data/);
  // console.log(v);
});

Deno.test("Iteration 2", () => {
  const d = new Dashboard(train);
  const v: string = d.render(2, [1, 2]);
  console.log(v);
});

Deno.test("Iteration 100", () => {
  const d = new Dashboard(train);
  const v: string = d.render(100, Array(100).fill(0).map(_=>Math.random()));
  console.log(v);
});

Deno.test("Run Iterations", {ignore: false}, () => {
  const d = new Dashboard(train, 60, 10);
  d.run(20000, 0.1, 100);
});
