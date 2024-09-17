import { assertEquals, assertInstanceOf } from "@std/assert";
import { Network } from "./network.ts";
import { Scatter } from "./scatter.ts";
import type { Inputs, Outputs } from "./train.ts";
import { Train } from "./train.ts";

const network = new Network(2).dense(3).lrelu.dense(1).sigmoid;

// XOR training set
const xs: Inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const ys: Outputs = [[0], [1], [1], [0]];

// Initialize object

Deno.test("Initialize", () => {
  const s = new Scatter(network, xs, ys);
  console.log(s);
  assertInstanceOf(s, Scatter);
});

Deno.test("Plot", () => {
  const s = new Scatter(network, xs, ys);
  const printable = s.plot();
  console.log(printable);
});


// Deno.test("Autodetect ranges", () => {
//   const s = new Scatter(network, xs, ys);
//   assertEquals(s.xmin, 0);
//   assertEquals(s.xmax, 1);
//   assertEquals(s.ymin, 0);
//   assertEquals(s.ymax, 1);
// });

// Deno.test("Manually set ranges", () => {
//   const s = new Scatter(network, xs, ys, {xmin: -1, xmax: 2, ymin: -2, ymax: 3});
//   assertEquals(s.xmin, -1);
//   assertEquals(s.xmax, 2);
//   assertEquals(s.ymin, -2);
//   assertEquals(s.ymax, 3);
// });


// Train network
const train = new Train(network, xs, ys);
// train.epsilon = 0.001;
// train.run(20000, 0.9);
// console.log(network.export);

// Deno.test("Initialize", () => {
//   const s = new ScatterPlot(network, xs, ys);
//   assertInstanceOf(s, ScatterPlot);
// });

// Deno.test("Pixels", () => {
//   const s = new ScatterPlot(network, xs, ys);
//   const size = 4;
//   const cols = size * 2;
//   const rows = size * 2;
//   const p: Uint8Array = s.pixels(cols, rows);

//   const image = new PixMap(cols, rows);
//   for (let col = 0; col < cols; ++col) {
//     for (let row = 0; row < rows; ++row) {
//       const index = (row * cols + col) * 4;
//       const [r, g, b] = [
//         p[index],
//         p[index+1],
//         p[index+2],
//       ];
//       image.set(col, row, new Color(r, g, b));
//     }
//   }

//   // Display blocks
//   console.log(image.toString());

//   assertInstanceOf(p, Uint8Array);
//   assertEquals(p.length, size ** 2 * 4 * 4);
//   const red = new Uint8Array([64, 0, 0, 255]);
//   const green = new Uint8Array([0, 255, 0, 255]);
//   // const topLeftIndex = 0;
//   const line = size * 2 * 4;
//   const topLeftIndex = line;
//   const topRightIndex = line + line - 4;
//   const bottomLeftIndex = size * 2 * line - line;
//   const bottomRightIndex = size * 2 * line - 4;

//   assertEquals(p.slice(topLeftIndex, topLeftIndex + 4), green);
//   assertEquals(p.slice(topRightIndex, topRightIndex + 4), red);
//   assertEquals(p.slice(bottomLeftIndex, bottomLeftIndex + 4), red);
//   assertEquals(p.slice(bottomRightIndex, bottomRightIndex + 4), green);
// });
