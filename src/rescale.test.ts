import { assertEquals, assertInstanceOf } from "@std/assert";
import { Rescale, Simple } from "./layer.ts";
import { Value } from "./value.ts";
import { Network } from "./network.ts";
import { Train } from "./train.ts";

Deno.test("Instance", ()=>{
  const r = new Rescale(1);
  assertInstanceOf(r, Rescale);
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
