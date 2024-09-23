import { assertEquals, assertInstanceOf } from "@std/assert";
import { Column, Training } from "./data.ts";
import { xor } from "../examples.ts";

Deno.test("Column Instance", () => {
  const c = new Column([]);
  assertInstanceOf(c, Column);
});

Deno.test("Min, Max, Mean", () => {
  const c = new Column([3, 1, 2]);
  assertEquals(c.min, 1);
  assertEquals(c.max, 3);
  assertEquals(c.mean, 2);
});

Deno.test("Distribution of points", () => {
  const c = new Column([3, 1, 2]);
  const p: number[] = c.points(5);
  assertEquals(p, [1, 1.5, 2, 2.5, 3]);
});

const example = xor();

Deno.test("Instance", () => {
  const t = new Training(example.inputs, example.outputs, 0, 1, 0);
  assertInstanceOf(t, Training);
});

Deno.test("Scatter", () => {
  const t = new Training(example.inputs, example.outputs, 0, 1, 0);
  const scatter = t.scatter(3, 3);
  assertEquals(scatter, [
    [0, 0],   [0.5, 0],   [1, 0],
    [0, 0.5], [0.5, 0.5], [1, 0.5],
    [0, 1],   [0.5, 1],   [1, 1],
  ]);
});
