//  1 161 168
//  2 156 163
//  3 151 158
//  4 146 153
//  5 141 148
//  6 137 144
//  7 132 139
//  8 127 134
//  9 122 129
// 10 118 124
// 11 113 120
// 12 108 115



import type { Network } from "./network.ts";
import { Training } from "./scatter/data.ts";
import type { Inputs, Outputs, Column } from "./scatter/data.ts";
import { Prediction } from "./scatter/prediction.ts";
import { blockify } from "image";

// Visualize training results

// /** Middle of sorted values */
// function mean(n: number[]): number {
//   const sorted = n.slice().sort();
//   const middle = Math.floor(sorted.length / 2);
//   return sorted[middle];
// }

// /** Extract column from grid */
// function column(grid: DataSet, index: number): number[] {
//   return grid.map((row) => row[index]);
// }

// /** Min and max number in array */
// function MinMax(array: number[]): [number, number] {
//   return [Math.min(...array), Math.max(...array)];
// }

// /** Convert number in input range to number in output range */
// function scale(
//   input_min: number,
//   input_max: number,
//   output_min: number,
//   output_max: number,
//   input: number
// ): number {
//   return (
//     ((input - input_min) / (input_max - input_min)) *
//       (output_max - output_min) +
//     output_min
//   );
// }

// type Row = number[];
// type DataSet = Array<Row>;

/** Generate a string of consecutive chars */
function pad(width: number, char: string = " "): string {
  return new Array(width).fill(char).join("");
}

/** Extract column from DataSet */
// function column(grid: DataSet, n: number): Column {
//   return new Column(grid.map((row: Row) => row[n]));
// }

// type Color = [number, number, number, number]; // R, G, B, A

// /** A canvas in range xmin:xmax, ymin:ymax */
// class PixelCanvas {
//   public readonly buffer: Uint8Array;

//   constructor(
//     private readonly xmin: number,
//     private readonly xmax: number,
//     private readonly ymin: number,
//     private readonly ymax: number,
//     private readonly xsize: number,
//     private readonly ysize: number
//   ) {
//     this.buffer = new Uint8Array(Array(xsize * ysize * 4).fill(0));
//   }

//   /** Convert floating (x,y) position to buffer index */
//   // TODO: Odd lines are ignored in terminal output
//   private pos(x: number, y: number): number {
//     const xg: number = Math.round(
//       scale(this.xmin, this.xmax, 0, this.xsize - 1, x)
//     );
//     const yg: number = Math.round(
//       scale(this.ymin, this.ymax, 0, this.ysize - 1, y)
//     );
//     // const yg: number = 2*Math.floor(
//     //   scale(this.ymin, this.ymax, 0, this.ysize - 1, y)/2,
//     // );
//     return ((this.ysize - yg - 1) * this.xsize + xg) * 4;
//   }

//   /** Set value at position */
//   public set(x: number, y: number, color: Color): void {
//     const offset = this.pos(x, y);
//     // console.log({ buffer: this.buffer, x, y, offset, color });
//     this.buffer.set(color, offset);
//   }
// }

// export class ScatterPlot {
//   /**
//    * @param   {Network}  network  Neural network
//    * @param   {DataSet}  xs       Training input values
//    * @param   {DataSet}  ys       Training output values
//    */
//   constructor(
//     private readonly network: Network,
//     private readonly xs: DataSet,
//     private readonly ys: DataSet
//   ) {}

//   /**
//    * Create a pixbuffer of the plot
//    *
//    * @param   {number}      xsize  Number of columns
//    * @param   {number}      ysize  Number of rows, default=xsize
//    * @param   {number}      xcol   Input column number for x-axis, default=0
//    * @param   {number}      ycol   Input column number for y-axis, default=1
//    * @param   {number}      vcol   Output column number for values, default=0
//    * @return  {Uint8Array}         8-bit RGBT buffer
//    */
//   public pixels(
//     xsize = 16,
//     ysize = xsize,
//     xcol = 0,
//     ycol = 1,
//     vcol = 0
//   ): Uint8Array {
//     // Identify data columns
//     const xs: number[] = column(this.xs, xcol);
//     const ys: number[] = column(this.xs, ycol);
//     const vs: number[] = column(this.ys, vcol);

//     // Identify minimum and maximum in each column
//     const xmin: number = Math.min(...xs);
//     const xmax: number = Math.max(...xs);
//     const ymin: number = Math.min(...ys);
//     const ymax: number = Math.max(...ys);
//     const vmin: number = Math.min(...vs);
//     const vmax: number = Math.max(...vs);

//     // Identify mean value in all input columns
//     // TODO: Skip columns used in grid
//     const means: number[] = this.xs[0].map((_, i) => mean(column(this.xs, i)));

//     // Create list of predicted values at each grid position
//     const values: Array<[number, number, number]> = [];
//     const xstep: number = (xmax - xmin) / (xsize - 1);
//     const ystep: number = (xmax - xmin) / (ysize - 1);
//     // console.log({xmin, xmax, xstep, ymin, ymax, ystep});
//     for (let x = xmin; x <= xmax; x += xstep) {
//       for (let y = ymin; y <= ymax; y += ystep) {
//         const input: number[] = [...means];
//         input[xcol] = x;
//         input[ycol] = y;
//         const output = this.network.predict(input);
//         const value = output[vcol];
//         values.push([x, y, value]);
//       }
//     }

//     // Plot predicted values (contour plot)

//     const canvas = new PixelCanvas(xmin, xmax, ymin, ymax, xsize, ysize);
//     let [pmin, pmax] = MinMax(column(values, 2));
//     if (vmin < pmin) pmin = vmin;
//     if (vmax > pmax) pmax = vmax;
//     values.forEach(([x, y, p]) => {
//       const lightness: number = Math.floor(scale(pmin, pmax, 0, 255, p));
//       // const color: Color = [255 - lightness, 128, 128, 255];
//       const color: Color = [lightness, lightness, lightness, 255];
//       canvas.set(x, y, color);
//     });

//     // Overlay training set (scatter plot)
//     this.xs.forEach((input, index) => {
//       const [x, y] = [input[xcol], input[ycol]];
//       const v: number = this.ys[index][vcol];
//       const lightness = Math.floor(scale(pmin, pmax, 0, 255, v));
//       const color: Color =
//         lightness >= 128
//           ? [0, lightness, 0, 255] // green
//           : [lightness + 64, 0, 0, 255]; // red
//       canvas.set(x, y, color);
//     });

//     return canvas.buffer;
//   }
// }

/** A column of numbers */
// class Column {
//   private readonly sorted: number[];

//   constructor(data: number[]) {
//     this.sorted = data.slice().sort();
//   }

//   /** Largest number */
//   public get max(): number {
//     return this.sorted[this.sorted.length - 1];
//   }

//   /** Smallest number */
//   public get min(): number {
//     return this.sorted[0];
//   }

//   /** Middle number */
//   public get mean(): number {
//     return this.sorted[Math.round(this.sorted.length / 2)];
//   }

//   /** Evenly distributed points between min and max, both included */
//   public points(count: number): number[] {
//     const low: number = this.min;
//     const high: number = this.max;
//     return Array(count)
//       .fill(0)
//       .map((_, i) => low + (i * (high - low)) / (count - 1));
//   }
// }

/** Pull predictions from neural network and scale to fit heatmap range */
// class Predictions {
//   /** Raw values from neural network */
//   private readonly predicted: number[][] = [];

//   constructor(
//     private readonly network: Network,
//     private readonly means: number[],
//     private readonly xs: number[],
//     private readonly ys: number[],
//     private readonly xcol: number,
//     private readonly ycol: number,
//     private readonly zcol: number,
//     public min: number,
//     public max: number
//   ) {
//     ys.forEach((y) => {
//       const row: number[] = [];
//       xs.forEach((x) => {
//         means[xcol] = x;
//         means[ycol] = y;
//         const output = network.predict(means);
//         const value: number = output[zcol];
//         if (value < this.min) this.min = value;
//         if (value > this.max) this.max = value;
//         row.push(value);
//       });
//       this.predicted.push(row);
//     });
//   }

//   // When looking up values, do scaling to a range
//   // But the range may extend to values from overlay
// }

class Heatmap {
  /** Row of mean values from all columns */
  private readonly prediction: Prediction;
  private readonly zcol: number;
  private readonly overlay: Array<[number, number, number]>;

  /**
   * @param {number} width Number of chars wide
   * @param {number} height Number of lines high
   * @param {Network} network Neural Network
   * @param {Training} train Inputs/Outputs data
   */
  constructor(
    private readonly width: number,
    private readonly height: number,
    private readonly network: Network,
    private readonly train: Training
  ) {
    const inputs = train.scatter(width * 2, height * 2);
    this.prediction = new Prediction(network, inputs);
    this.zcol = train.zcol;

    const xs: number[] = train.xs.data;
    const ys: number[] = train.ys.data;
    const zs: number[] = train.values.data;
    this.overlay = xs.map((x, i) => [x, ys[i], zs[i]]);
  }

  public min(): number {
    return Math.min(...this.values());
  }

  public max(): number {
    return Math.max(...this.values());
  }

  private values(): number[] {
    return this.prediction.outputs(this.zcol);
  }

  /** Scale all outputs to range 0-256 */
  private scaled(): number[] {
    const input_min = this.min();
    const input_max = this.max();
    const output_min = 0;
    const output_max = 256;
    return this.values().map(
      (input: number) =>
        ((input - input_min) / (input_max - input_min)) *
          (output_max - output_min) +
        output_min
    );
  }

  /** Render heatmap as lines of strings */
  public render(): string[] {
    // const values = this.prediction.outputs(this.zcol);
    const scaled: number[] = this.scaled();
    console.log({ scaled });
    const ints = scaled.map((f) => (f == 256 ? 255 : Math.floor(f)));
    console.log({ scaled, ints });
    const bitmap: Uint8Array = new Uint8Array(
      ints.map((i) => [i, i, i, 0]).flat()
    );
    const long: string = blockify(bitmap, this.width * 2, this.height * 2);
    const printable = long.replaceAll('\x1b[39m\x1b[49m\x1b', '\x1b');
    const lines = printable.split("\n");
    return lines;
    // TODO: Render real map
    // const x: string = pad(this.width, "x");
    // return new Array(this.height).fill(x);
  }
}

////////////////////////////////////////////////////////////////////////
/// X Axis
////////////////////////////////////////////////////////////////////////

/** Render values and label on X Axis */
class XAxis {
  /**
   * @param {string} name Label for X Axis
   * @param {number} high Highest value on X axis
   * @param {number} low Lowest value on X axis
   * @param {number} width Number of chars wide
   */
  constructor(
    private readonly name: string,
    private readonly high: number,
    private readonly low: number,
    private readonly width: number
  ) {}

  /** Number of lines high */
  public get height(): 1 {
    return 1;
  }

  /** Generate all lines in X Axis
   * TODO: Align to edges of heatmap
   * TODO: This output is static, so only render once
   */
  public render(): string[] {
    const low: string = this.low.toPrecision(2);
    const high: string = this.high.toPrecision(2);
    const labels: string = [low, this.name, high].join("  ");
    const padwidth: number = this.width - labels.length;
    console.log({ low, high, labels, width: this.width, padwidth });
    const padding: string = pad(this.width - labels.length);
    return [padding + labels];
  }
}

////////////////////////////////////////////////////////////////////////
/// Y Axis
////////////////////////////////////////////////////////////////////////

/** Render values and label on Y Axis */
class YAxis {
  /**
   * @param {string} name Label for Y Axis
   * @param {number} high Highest value on Y axis
   * @param {number} low Lowest value on Y axis
   * @param {number} height Number of lines
   */
  constructor(
    private readonly name: string,
    private readonly high: number,
    private readonly low: number,
    private readonly height: number
  ) {}

  /** Highest number as a string */
  private get highLabel(): string {
    return this.high.toPrecision(2);
  }

  /** Highest number as a string */
  private get lowLabel(): string {
    return this.low.toPrecision(2);
  }

  /** Max length of labels */
  public get width(): number {
    return Math.max(
      this.highLabel.length,
      this.name.length,
      this.lowLabel.length
    );
  }

  /** Generate all lines in Y Axis
   * TODO: This output is static, so only render once
   */
  public render(): string[] {
    const w: number = this.width; // Width
    const h: number = this.height; // Height
    const padding: string = pad(w);
    const lines: string[] = Array(h).fill(padding);
    lines[0] = padding.slice(0, w - this.highLabel.length) + this.highLabel;
    lines[Math.round((h - 1) / 2)] =
      padding.slice(0, w - this.name.length) + this.name;
    lines[h - 1] = padding.slice(0, w - this.lowLabel.length) + this.lowLabel;
    return lines;
  }
}

////////////////////////////////////////////////////////////////////////
/// Z Axis
////////////////////////////////////////////////////////////////////////

/** Render values and label on Z Axis */
class ZAxis {
  /**
   * @param {string} name Label for Z Axis
   * @param {number} width Number of chars wide
   */
  constructor(private readonly name: string, private readonly width: number) {}

  /** Number of lines high */
  public get height(): 1 {
    return 1;
  }

  /** Generate all lines in Z Axis
   * @param {number} low Lowest output value
   * @param {number} high Highest output value
   */
  public render(low: number, high: number): string[] {
    // TODO: Color encode to black background and white foreground
    const labels = "▯ " + low.toPrecision(2) + "  ▮ " + high.toPrecision(2);
    const padding: string = pad(this.width - labels.length);
    return [padding + labels];
  }
}

/** Traverse range of values for two parameters of the input set,
 * use mean input values for all other parameters,
 * generate heatmap for the values in range,
 * overlay values from output set,
 * and generate ANSI digram printable on console.
 */
export class Scatter {
  /** Column number in input to use at x-axis */
  public readonly xcol: number = 0;

  /** Label on x-axis */
  public readonly xlabel: string = "X";

  /** Column number in input to use at y-axis */
  public readonly ycol: number = 1;

  /** Label on y-axis */
  public readonly ylabel: string = "Y";

  /** Column number in output to use at values */
  public readonly zcol: number = 0;

  /** Label for values */
  public readonly zlabel: string = "Output";

  /** Total width of diagrams in chars */
  public readonly width: number = 40;

  /** Total height of diagrams in lines of chars */
  public readonly height: number = 11;

  // Components in diagram
  private readonly xaxis: XAxis;
  private readonly yaxis: YAxis;
  private readonly zaxis: ZAxis;
  private readonly heatmap: Heatmap;

  /**
   * @param {Network} network - Neural network
   * @param {DataSet} input   - Training input values, array of rows of numbers
   * @param {DataSet} output  - Training output values, array of rows of numbers
   * @param {Scatter} config  - Options for x, y and z axis
   */
  constructor(
    private readonly network: Network,
    private readonly input: Inputs,
    private readonly output: Outputs,
    private readonly config: Partial<Scatter> = {}
  ) {
    Object.assign(this, config);
    // this.setMeans();
    // this.setRanges();
    // this.setLayout();
    // this.setLeftBar();

    const data = new Training(input, output, this.xcol, this.ycol, this.zcol);

    const xs: Column = data.xs;
    const ys: Column = data.ys;
    this.xaxis = new XAxis(this.xlabel, xs.max, xs.min, this.width);
    this.zaxis = new ZAxis(this.zlabel, this.width);
    const contentHeight: number =
      this.height - this.zaxis.height - this.xaxis.height;
    this.yaxis = new YAxis(this.ylabel, ys.max, ys.min, contentHeight);
    this.heatmap = new Heatmap(
      this.width - this.yaxis.width,
      contentHeight,
      network,
      data
    );
  }

  /** If xmin, xmax, ymin, ymax is not provided, then autodect from input */
  // private setRanges(): void {
  //   // Define xmin and xmax
  //   if (this.xmin === undefined || this.xmax === undefined) {
  //     const xval: Column = this.input.map((row) => row[this.xcol]).sort();
  //     if (this.xmin === undefined) this.xmin = xval[0];
  //     if (this.xmax === undefined) this.xmax = xval[xval.length - 1];
  //   }

  //   // Define ymin and ymax
  //   if (this.ymin === undefined || this.ymax === undefined) {
  //     const yval: Column = this.input.map((row) => row[this.ycol]).sort();
  //     if (this.ymin === undefined) this.ymin = yval[0];
  //     if (this.ymax === undefined) this.ymax = yval[yval.length - 1];
  //   }

  //   // Reduce number of digits in labels
  //   this.xminLabel = this.xmin.toPrecision(2);
  //   this.xmaxLabel = this.xmax.toPrecision(2);
  //   this.yminLabel = this.ymin.toPrecision(2);
  //   this.ymaxLabel = this.ymax.toPrecision(2);
  // }

  /** Return middle value of sorted values */
  // private static mean(col: Column): number {
  //   const sorted = col.slice().sort();
  //   const middle: number = Math.round(sorted.length / 2);
  //   return sorted[middle];
  // }

  /** Define dimensions of areas in layout */
  // private setLayout(): void {
  //   this.leftBarWidth = Math.max(
  //     this.yminLabel.length,
  //     this.ymaxLabel.length,
  //     this.ylabel.length
  //   );
  //   this.leftBarHeight = this.height - this.topBarHeight - this.bottomBarHeight;
  //   this.heatmapWidth = this.width - this.leftBarWidth;
  //   this.heatmapHeight = this.height - this.topBarHeight - this.bottomBarHeight;
  // }

  /** Define lines in Left Bar */
  // private setLeftBar(): void {
  //   const w: number = this.leftBarWidth; // Width
  //   const h: number = this.leftBarHeight; // Height
  //   const padding: string = Array(w).fill(" ").join("");
  //   const lines: string[] = Array(h).fill(padding);
  //   lines[0] = padding.slice(0, w - this.ymaxLabel.length) + this.ymaxLabel;
  //   lines[Math.round((h - 1) / 2)] =
  //     padding.slice(0, w - this.ylabel.length) + this.ylabel;
  //   lines[h - 1] = padding.slice(0, w - this.yminLabel.length) + this.yminLabel;

  //   this.leftBar.push(...lines);
  // }

  /** Extract all values from one column */

  /** Get grid of values from Network */
  // private values(xcount: number, ycount: number): DataSet {
  //   const result: number[][] = [];
  //     return this._value;
  //     this._value = v;
  //   }

  //   for (
  //     let y = this.ymin;
  //     y <= this.ymax;
  //     y += (this.ymax - this.ymin) / (ycount - 1)
  //   ) {
  //     const row: Row = [];
  //     for (
  //       let x = this.xmin;
  //       x <= this.xmax;
  //       x += (this.xmax - this.xmin) / (xcount - 1)
  //     ) {
  //       const input = this.means;
  //       input[this.xcol] = x;
  //       input[this.ycol] = y;
  //       const output: Row = this.network.predict(input);
  //       const value: number = output[this.zcol];
  //       row.push(value);
  //     }
  //     result.push(row);
  //   }
  //   return result;
  // }

  /** Generate the unannotated heatmap */
  // private heatmap(width: number, height: number): string {
  //   return "";
  // }

  /** Diagram Layout:
   * ┌Header────────────────────────────────┐
   * │   ▯ Lowest output  ▮ Highest output │
   * └──────────────────────────────────────┘
   * ┌LeftBar───┐┌Content───────────────────┐
   * │ Highest_Y││┌────────────────────────┐│
   * │          │││                        ││
   * │Y_Axisname│││        heatmap         ││
   * │          │││                        ││
   * │  Lowest_Y││└────────────────────────┘│
   * └──────────┘└──────────────────────────┘
   * ┌Footer────────────────────────────────┐
   * │        Lowest_X  Y_Axisname  Lowest_Y│
   * └──────────────────────────────────────┘
   */
  // private layout(): number {
  //   // Reserved space for labels on Y axis
  //   // TODO: Calculate max of [Lowest_Y, Highest_Y and Y_Axisname]
  //   return 2;
  // }

  /** Generate heatmap diagram
   * @param [width=40] Number of chars wide
   * @param [height=10] Number of chars high
   * @result String printable on terminal console
   */
  public plot(): string {
    const header = this.zaxis.render(this.heatmap.min(), this.heatmap.max());
    const yaxis = this.yaxis.render();
    const heatmap = this.heatmap.render();
    const footer = this.xaxis.render();

    const lines: string[] = [
      ...header,
      ...yaxis.map((line, index) => [line, heatmap[index]].join("")),
      ...footer,
    ];

    return lines.join("\n");
  }
}
