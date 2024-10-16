import { assertInstanceOf, assertNotEquals } from "@std/assert";
import { Neuron, Normalizer } from "./neuron.ts";
import { Value } from "./value.ts";

Deno.test("Neuron Instance", () => {
  const n = new Neuron(0);
  assertInstanceOf(n, Neuron);
});

Deno.test("Neuron Activation", () => {
  const n = new Neuron(1);
  const o = n.forward([new Value(1)]);
  assertNotEquals(o.data, 0);
});

Deno.test("Normalizer Instance", () => {
  const n = new Normalizer();
  assertInstanceOf(n, Normalizer);
});

Deno.test("Normalizer activation", () => {
  const n = new Normalizer();
  const o = n.forward(new Value(1));
  assertNotEquals(o.data, 0);
});
