import { assertEquals, assertInstanceOf } from "@std/assert";
import { Network } from "./network.ts";
import { Scatter } from "./scatter.ts";
import type { Inputs, Outputs } from "./train.ts";
import { Train } from "./train.ts";
import { plot } from "chart";

const xor = new Network(2).dense(3).lrelu.dense(1).sigmoid;

// XOR training set
const xs: Inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const ys: Outputs = [[0], [1], [1], [0]];

// Resample data to 80 columns and display ascii chart
function plot_graph(data: number[], height: number): string {
  const step = (data.length - 1) / 26;
  const samples: number[] = [];
  for (let i = 0; i < data.length; i += step) {
    samples.push(data[Math.floor(i)]);
  }
  // console.log({ samples });
  return plot(samples, { height: height - 1, padding: "     " });
}

Deno.test("Initialize", () => {
  const s = new Scatter(xor, xs, ys);
  // console.log(s);
  assertInstanceOf(s, Scatter);
});

Deno.test("Untrained and trained plot", () => {
  const s = new Scatter(xor, xs, ys);
  const initial = s.plot();
  console.log(initial);

  const train = new Train(xor, xs, ys);
  train.epsilon = 0.001;
  train.run(20000, 0.9);

  const trained = s.plot();
  console.log(trained);
});

Deno.test("Circle Training", () => {
  // Create network
  const circle = new Network(2).dense(16).lrelu.dense(8).lrelu.dense(1).tanh;

  // Generate test data for a fat circle
  const xs: Inputs = [];
  const ys: Outputs = [];
  for (let i = 0; i < 150; ++i) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    const r = Math.sqrt(x * x + y * y);
    // Circle
    const c = r >= 0.45 && r <= 0.75 ? 1 : -1;
    // Wave
    const w = r < 0.5 ? 2 * r - 0.5 : 1.5 - 2 * r;
    xs.push([x, y]);
    ys.push([w]);
  }
  // console.log({xs, ys})

  const s = new Scatter(circle, xs, ys);
  // const initial = s.plot();
  // console.log(initial);

  const train = new Train(circle, xs, ys);
  train.epsilon = 0.001;
  train.callbackFrequency = 100;
    let first = true;
  train.callback = (iterations: number, losses: number[]) => {
    // console.log("\u001bc"); // Clear screen
    // console.log("\u001B[H"); // Home
    // console.log(s.plot());
    // console.log(plot_graph(losses, 11));

    if (!first) {
      // console.log("Going up");
      console.log("\u001B[14F");
    } else first = false;

    const scatter: string[] = s.plot().split("\n");
    const loss: string[] = plot_graph(losses, 11).split("\n");

    scatter.forEach((line, index) => {
      console.log(line + loss[index]);
    });

    console.log(`Iterations: ${iterations}\n`);
  };
  // console.log("\u001bc"); // Clear screen
  console.log("\u001B[?25l"); // Hide cursor
  train.run(20000, 0.4);
  console.log("\u001B[?25h"); // Show cursor

  // const trained = s.plot();
  // console.log(trained);
});
