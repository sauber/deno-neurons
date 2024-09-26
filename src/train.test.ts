import { assertAlmostEquals, assertInstanceOf } from "@std/assert";
import { Network } from "./network.ts";
import { Train } from "./train.ts";
import { Value } from "./value.ts";
import { xor } from "./examples.ts";


Deno.test("Initialize", () => {
  const network = new Network(0);
  const train = new Train(network, [], []);
  assertInstanceOf(train, Train);
});

// In theory Dense(2), Sigmoid, Dense(1) should be enough,
// but takes a long time to train, and often doesn't find the solution.
// A much larger network using relu is faster and higher chance of success.
Deno.test("XOR training", () => {
  const ex = xor();

  const train = new Train(ex.network, ex.inputs, ex.outputs);
  train.epsilon = 0.01;
  train.run(200000, 0.9);

  // Validate
  ex.inputs.forEach((x, i) => {
    const p = ex.network.forward(x.map((v) => new Value(v))).map((v) => v.data);
    console.log(x, p);
    assertAlmostEquals(p[0], ex.outputs[i][0], 0.3);
  });
});
