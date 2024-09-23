import { assertInstanceOf, assertLessOrEqual, assertMatch, assertNotMatch } from "@std/assert";
import { Dashboard } from "./dashboard.ts";
import { Train } from "./train.ts";
import { xor } from "./examples.ts";
import type { Example } from "./examples.ts";

const x: Example = xor();
const train = new Train(x.network, x.inputs, x.outputs);
train.epsilon = 0.01;

Deno.test("Instance", () => {
  const d = new Dashboard(train);
  assertInstanceOf(d, Dashboard);
});

Deno.test("No iterations", () => {
  const d = new Dashboard(train);
  const v: string = d.render(0, []);
  assertMatch(v, /No data/);
});

Deno.test("Iteration 1", () => {
  const d = new Dashboard(train);
  const v: string = d.render(1, [1]);
  assertMatch(v, /No data/);
});

Deno.test("Iteration 2", () => {
  const d = new Dashboard(train);
  const v: string = d.render(2, [1, 2]);
  assertNotMatch(v, /No data/);
});

Deno.test("Iteration 100", () => {
  const d = new Dashboard(train);
  const v: string = d.render(
    100,
    Array(100)
      .fill(0)
      .map((_) => Math.random())
  );
  assertNotMatch(v, /No data/);
});

Deno.test("Run Iterations", { ignore: false }, () => {
  const d = new Dashboard(train, 60, 10);
  const max = 2000;
  const iterations = d.run(max, 0.4, 10);
  assertLessOrEqual(iterations, max);
});
