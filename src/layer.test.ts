import {
  assertAlmostEquals,
  assertEquals,
  assertInstanceOf,
} from "@std/assert";
import {
  Dense,
  LRelu,
  Normalize,
  Relu,
  Sigmoid,
  Simple,
  Tanh,
} from "./layer.ts";
import { Network } from "./network.ts";
import type { NetworkData } from "./network.ts";
import type { DenseData } from "./layer.ts";
import type { Inputs } from "./train.ts";
import { v } from "./value.ts";
import { avg, std } from "@sauber/statistics";

////////////////////////////////////////////////////////////////////////
/// Relu Layer
////////////////////////////////////////////////////////////////////////

Deno.test("Relu Instance", () => {
  assertInstanceOf(new Relu(), Relu);
});

Deno.test("Relu Import/export", () => {
  const n = new Network(0).relu;
  const e: NetworkData = n.export;
  assertEquals(e, { inputs: 0, layers: ["Relu"] });
  const i = Network.import(e);
  assertInstanceOf(i, Network);
});

Deno.test("Relu activation", () => {
  const n = new Network(1).relu;
  const o = n.predict([0]);
  assertEquals(o, [0]);
});

////////////////////////////////////////////////////////////////////////
/// Leaky Relu Layer
////////////////////////////////////////////////////////////////////////

Deno.test("Leaky-Relu Instance", () => {
  assertInstanceOf(new LRelu(), LRelu);
});

Deno.test("Leaky-Relu Import/export", () => {
  const n = new Network(0).lrelu;
  const e: NetworkData = n.export;
  assertEquals(e, { inputs: 0, layers: ["LRelu"] });
  const i = Network.import(e);
  assertInstanceOf(i, Network);
});

Deno.test("Leaky-Relu activation", () => {
  const n = new Network(1).lrelu;
  const o = n.predict([-1]);
  assertEquals(o, [-0.01]);
});

////////////////////////////////////////////////////////////////////////
/// Sigmoid Layer
////////////////////////////////////////////////////////////////////////

Deno.test("Sigmoid Instance", () => {
  assertInstanceOf(new Sigmoid(), Sigmoid);
});

Deno.test("Sigmoid Import/export", () => {
  const n = new Network(0).sigmoid;
  const e: NetworkData = n.export;
  assertEquals(e, { inputs: 0, layers: ["Sigmoid"] });
  const i = Network.import(e);
  assertInstanceOf(i, Network);
});

Deno.test("Sigmoid activation", () => {
  const n = new Network(1).sigmoid;
  const o = n.predict([0]);
  assertEquals(o, [0.5]);
});

////////////////////////////////////////////////////////////////////////
/// Tanh Layer
////////////////////////////////////////////////////////////////////////

Deno.test("Tanh Instance", () => {
  assertInstanceOf(new Tanh(), Tanh);
});

Deno.test("Tanh Import/export", () => {
  const n = new Network(0).tanh;
  const e: NetworkData = n.export;
  assertEquals(e, { inputs: 0, layers: ["Tanh"] });
  const i = Network.import(e);
  assertInstanceOf(i, Network);
});

Deno.test("Tanh activation", () => {
  const n = new Network(1).tanh;
  const o = n.predict([0]);
  assertEquals(o, [0]);
});

////////////////////////////////////////////////////////////////////////
/// Simple Layer
////////////////////////////////////////////////////////////////////////

Deno.test("Simple Instance", () => {
  assertInstanceOf(new Simple(0), Simple);
});

Deno.test("Simple Import/export", () => {
  const n = new Network(0).simple;
  const e: NetworkData = n.export;
  assertEquals(e, { inputs: 0, layers: [{ Simple: [] }] });
  const i = Network.import(e);
  assertInstanceOf(i, Network);
});

Deno.test("Simple activation", () => {
  const data: NetworkData = {
    inputs: 1,
    layers: [{ Simple: [{ bias: 0, weights: [0] }] }],
  };
  const n = Network.import(data);
  const o = n.predict([0]);
  assertEquals(o, [0]);
});

////////////////////////////////////////////////////////////////////////
/// Dense Layer
////////////////////////////////////////////////////////////////////////

Deno.test("Dense Instance", () => {
  assertInstanceOf(new Dense(0, 0), Dense);
});

Deno.test("Dense Import/export", () => {
  const cases = [0, 1, 2];
  cases.forEach((width) => {
    const n = new Network(width).dense(width);
    const e: NetworkData = n.export;
    assertEquals(e.layers.length, 1);
    const neuron = e.layers[0] as Record<"Dense", DenseData>;
    assertEquals(neuron.Dense.length, width);
    if (width > 0) assertEquals(neuron.Dense[0].weights.length, width);
    const i = Network.import(e);
    assertInstanceOf(i, Network);
  });
});

Deno.test("Dense activation", () => {
  const data: NetworkData = {
    inputs: 1,
    layers: [{ Dense: [{ bias: 0, weights: [0] }] }],
  };
  const n = Network.import(data);
  const o = n.predict([0]);
  assertEquals(o, [0]);
});

////////////////////////////////////////////////////////////////////////
/// Normalization Processing Layer
////////////////////////////////////////////////////////////////////////

Deno.test("Normalize Instance", () => {
  assertInstanceOf(new Normalize(0), Normalize);
});

Deno.test("Normalize Layer Adapt", () => {
  const input: Inputs = [
    [0.1, 1400],
    [0.01, 1300],
    [0.055, 1350],
  ];

  // Adapt, export and import
  const layer = new Normalize(0);
  layer.adapt(input);
  const e = layer.export;
  const l = Normalize.import(e);

  // Confirm stddev of input after adaption
  const output = input.map((i) =>
    l.forward(i.map((n) => v(n))).map((o) => o.data)
  );

  // Confirm column by column mean==0, variance==1
  output[0].forEach((_, index) => {
    const col = output.map((r) => r[index]);
    const mean = avg(col);
    const variance = std(col);
    assertAlmostEquals(mean, 0, 1e-15);
    assertAlmostEquals(variance, 1, 1e-15);
  });
});

Deno.test("Normalize Network Adapt", () => {
  const input: Inputs = [
    [0.1, 1400],
    [0.01, 1300],
    [0.055, 1350],
  ];

  // Adapt, export and import
  const network = new Network(2).normalize;
  network.adapt(input);

  // Confirm stddev of input after adaption
  const output = input.map((i) => network.predict(i));

  // Confirm column by column mean==0, variance==1
  output[0].forEach((_, index) => {
    const col = output.map((r) => r[index]);
    const mean = avg(col);
    const variance = std(col);
    assertAlmostEquals(mean, 0, 1e-15);
    assertAlmostEquals(variance, 1, 1e-15);
  });
});
