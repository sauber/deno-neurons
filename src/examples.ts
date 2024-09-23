import { Network } from "./network.ts";
import type { Inputs, Outputs } from "./train.ts";

/** A network to solve inputs to outputs */
export type Example = {
  network: Network;
  inputs: Inputs;
  outputs: Outputs;
};

/** XOR logic */
export function xor(): Example {
  return {
    network: new Network(2).dense(5).lrelu.dense(1).sigmoid,
    inputs: [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ],
    outputs: [[0], [1], [1], [0]],
  };
}

/** Sinus wave from middle of area */
export function wave(): Example {
  const xs: Inputs = [];
  const ys: Outputs = [];

  for (let i = 0; i < 150; ++i) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    const r = Math.sqrt(x * x + y * y);
    const s = -Math.sin(r * 7);
    xs.push([x, y]);
    ys.push([s]);
  }

  return {
    network: new Network(2).dense(8).lrelu.dense(8).lrelu.dense(1).tanh,
    inputs: xs,
    outputs: ys,
  };
}
