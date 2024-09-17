import {
  assertEquals,
  assertGreater,
  assertInstanceOf,
  assertLess,
} from "@std/assert";
import { Prediction } from "./prediction.ts";
import { Network } from "../network.ts";
import type { Inputs } from "./data.ts";

const network = new Network(2).dense(1).sigmoid;
const xs: Inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

Deno.test("Instance", () => {
  assertInstanceOf(new Prediction(network, []), Prediction);
});

Deno.test("Predictions", () => {
  const p = new Prediction(network, xs);
  const outputs: number[] = p.outputs(0);
  assertEquals(outputs.length, xs.length);
  outputs.forEach((v) => {
    assertGreater(v, 0.0);
    assertLess(v, 1.0);
  });
});
