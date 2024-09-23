import { assertGreater, assertInstanceOf } from "@std/assert";
import { Scatter } from "./scatter.ts";
import { Train } from "../train.ts";
import { xor, wave } from "../examples.ts";
import type { Example } from "../examples.ts";

Deno.test("Initialize", () => {
  const x: Example = xor();
  const s = new Scatter(x.network, x.inputs, x.outputs);
  assertInstanceOf(s, Scatter);
});

Deno.test("Untrained and trained plot", () => {
  const x: Example = xor();
  const s = new Scatter(x.network, x.inputs, x.outputs);
  const initial: string = s.plot();
  console.log(initial);

  const train = new Train(x.network, x.inputs, x.outputs);
  train.epsilon = 0.001;
  train.run(2000, 0.9);

  const trained = s.plot();
  console.log(trained);
});

Deno.test("Circle Training", () => {
  const w: Example = wave();
  const s = new Scatter(w.network, w.inputs, w.outputs);
  const train = new Train(w.network, w.inputs, w.outputs);
  console.log(s.plot());
  train.epsilon = 0.001;
  const iterations: number = train.run(2000, 0.4);

  const trained = s.plot();
  console.log(trained);
  console.log({iterations})
  assertGreater(iterations, 1);
});
