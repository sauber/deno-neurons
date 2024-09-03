import { assertAlmostEquals, assertInstanceOf } from "@std/assert";
import { Network } from "./network.ts";
import type { Inputs, Outputs } from "./train.ts";
import { Train } from "./train.ts";
import { Value } from "./value.ts";

// XOR training set
const xs: Inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const ys: Outputs = [[0], [1], [1], [0]];

Deno.test("Initialize", () => {
  const network = new Network(0);
  const train = new Train(network, [], []);
  assertInstanceOf(train, Train);
});

// In theory Dense(2), Sigmoid, Dense(1) should be enough,
// but takes a long time to train, and often doesn't find the solution.
// A much larger network using relu is faster and higher chance of success.
Deno.test("XOR training", () => {
  const network: Network = new Network(2)
    .dense(5)
    .lrelu.dense(2)
    .lrelu.dense(1).sigmoid;

  const train = new Train(network, xs, ys);
  train.epsilon = 0.01;
  train.run(200000, 0.9);

  // Validate
  xs.forEach((x, i) => {
    const p = network.forward(x.map((v) => new Value(v))).map((v) => v.data);
    console.log(x, p);
    assertAlmostEquals(p[0], ys[i][0], 0.2);
  });
});
