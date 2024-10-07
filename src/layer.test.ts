import { assertEquals } from "@std/assert";
import { Relu, Simple } from "./layer.ts";
import { Value } from "./value.ts";

Deno.test("Rely Activation Layer", () => {
  const l = new Relu();
  const cases = [
    [-1, 0],
    [0, 0],
    [1, 1],
  ];
  cases.forEach((c) => {
    const predict = l.forward([new Value(c[0])]);
    assertEquals(predict[0].data, c[1]);
  });
});

Deno.test("Simple Layer", () => {
  const s = new Simple(2);
  assertEquals(s.parameters.length, 4);
  assertEquals(s.export.length, 2);
  const predict = s.forward([0, 0].map((v) => new Value(v)));
  assertEquals(predict.length, 2);
});
