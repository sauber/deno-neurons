import { assertEquals, assertInstanceOf } from "@std/assert";
import { Rescale, Simple } from "./layer.ts";
import { Value, v } from "./value.ts";
import { Network } from "./network.ts";
import { MeanSquareError, Train } from "./train.ts";
import { Neuron } from "./neuron.ts";

Deno.test("Instance", () => {
  const r = new Rescale(1);
  assertInstanceOf(r, Rescale);
});

Deno.test("Scale Neuron", () => {
  const network = new Network(1).rescale.simple;
  console.log(network.export);

  const input = v(1400);
  const expected = v(1);
  for (let i = 0; i < 10; i++) {
    const p = network.forward([input]);
    // console.log(p);
    // p.backward();
    
    const loss = MeanSquareError([[expected]], [p]);
    console.log({ input: input.data, predict: p[0].data, expect: expected.data, loss: loss.data });
    // const loss = p.sub(input);
    // console.log(loss);
    // loss.print();

    network.zeroGrad();
    loss.backward();

    // loss.print();

    const learning_rate = 0.15;
    // console.log('parameters', network.parameters);
    for (const p of network.parameters) {
      // console.log(p);
      p.data -= learning_rate * p.grad;
    }

    // loss.print();

    // const q = network.forward([v(1)]);
    // console.log({predict: q.data, expect: ys.data});
  }
});

// Deno.test("Training", () => {
//   const s = new Network(1).rescale;
//   const inputs = [[0], [10]];
//   const outputs = [[0], [1]];
//   const t = new Train(s, inputs, outputs);
//   t.callback = (iteration: number, loss: number[]) => console.log({iteration, loss});
//   const r = t.run(20);
//   const l = t
//   console.log(r);
//   console.log(s.export);
// });
