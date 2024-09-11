import { Network } from "./network.ts";

// Visualize training results

/** Middle of sorted values */
function mean(n: number[]): number {
  const sorted = n.slice().sort();
  const middle = Math.floor(sorted.length / 2);
  return sorted[middle];
}

/** Extract column from grid */
function column(grid: DataSet, index: number): number[] {
  return grid.map((row) => row[index]);
}

/** Min and max number in array */
function MinMax(array: number[]): [number, number] {
  return [Math.min(...array), Math.max(...array)];
}

/** Convert number in input range to number in output range */
function scale(
  input_min: number,
  input_max: number,
  output_min: number,
  output_max: number,
  input: number
): number {
  return (
    ((input - input_min) / (input_max - input_min)) *
      (output_max - output_min) +
    output_min
  );
}

type Row = number[];
type Column = number[];
type DataSet = Array<Row>;
type Color = [number, number, number, number]; // R, G, B, A

/** A canvas in range xmin:xmax, ymin:ymax */
class PixelCanvas {
  public readonly buffer: Uint8Array;

  constructor(
    private readonly xmin: number,
    private readonly xmax: number,
    private readonly ymin: number,
    private readonly ymax: number,
    private readonly xsize: number,
    private readonly ysize: number
  ) {
    this.buffer = new Uint8Array(Array(xsize * ysize * 4).fill(0));
  }

  /** Convert floating (x,y) position to buffer index */
  // TODO: Odd lines are ignored in terminal output
  private pos(x: number, y: number): number {
    const xg: number = Math.round(
      scale(this.xmin, this.xmax, 0, this.xsize - 1, x)
    );
    const yg: number = Math.round(
      scale(this.ymin, this.ymax, 0, this.ysize - 1, y)
    );
    // const yg: number = 2*Math.floor(
    //   scale(this.ymin, this.ymax, 0, this.ysize - 1, y)/2,
    // );
    return ((this.ysize - yg - 1) * this.xsize + xg) * 4;
  }

  /** Set value at position */
  public set(x: number, y: number, color: Color): void {
    const offset = this.pos(x, y);
    // console.log({ buffer: this.buffer, x, y, offset, color });
    this.buffer.set(color, offset);
  }
}

export class ScatterPlot {
  /**
   * @param   {Network}  network  Neural network
   * @param   {DataSet}  xs       Training input values
   * @param   {DataSet}  ys       Training output values
   */
  constructor(
    private readonly network: Network,
    private readonly xs: DataSet,
    private readonly ys: DataSet
  ) {}

  /**
   * Create a pixbuffer of the plot
   *
   * @param   {number}      xsize  Number of columns
   * @param   {number}      ysize  Number of rows, default=xsize
   * @param   {number}      xcol   Input column number for x-axis, default=0
   * @param   {number}      ycol   Input column number for y-axis, default=1
   * @param   {number}      vcol   Output column number for values, default=0
   * @return  {Uint8Array}         8-bit RGBT buffer
   */
  public pixels(
    xsize = 16,
    ysize = xsize,
    xcol = 0,
    ycol = 1,
    vcol = 0
  ): Uint8Array {
    // Identify data columns
    const xs: number[] = column(this.xs, xcol);
    const ys: number[] = column(this.xs, ycol);
    const vs: number[] = column(this.ys, vcol);

    // Identify minimum and maximum in each column
    const xmin: number = Math.min(...xs);
    const xmax: number = Math.max(...xs);
    const ymin: number = Math.min(...ys);
    const ymax: number = Math.max(...ys);
    const vmin: number = Math.min(...vs);
    const vmax: number = Math.max(...vs);

    // Identify mean value in all input columns
    // TODO: Skip columns used in grid
    const means: number[] = this.xs[0].map((_, i) => mean(column(this.xs, i)));

    // Create list of predicted values at each grid position
    const values: Array<[number, number, number]> = [];
    const xstep: number = (xmax - xmin) / (xsize - 1);
    const ystep: number = (xmax - xmin) / (ysize - 1);
    // console.log({xmin, xmax, xstep, ymin, ymax, ystep});
    for (let x = xmin; x <= xmax; x += xstep) {
      for (let y = ymin; y <= ymax; y += ystep) {
        const input: number[] = [...means];
        input[xcol] = x;
        input[ycol] = y;
        const output = this.network.predict(input);
        const value = output[vcol];
        values.push([x, y, value]);
      }
    }

    // Plot predicted values (contour plot)

    const canvas = new PixelCanvas(xmin, xmax, ymin, ymax, xsize, ysize);
    let [pmin, pmax] = MinMax(column(values, 2));
    if (vmin < pmin) pmin = vmin;
    if (vmax > pmax) pmax = vmax;
    values.forEach(([x, y, p]) => {
      const lightness: number = Math.floor(scale(pmin, pmax, 0, 255, p));
      // const color: Color = [255 - lightness, 128, 128, 255];
      const color: Color = [lightness, lightness, lightness, 255];
      canvas.set(x, y, color);
    });

    // Overlay training set (scatter plot)
    this.xs.forEach((input, index) => {
      const [x, y] = [input[xcol], input[ycol]];
      const v: number = this.ys[index][vcol];
      const lightness = Math.floor(scale(pmin, pmax, 0, 255, v));
      const color: Color =
        lightness >= 128
          ? [0, lightness, 0, 255] // green
          : [lightness + 64, 0, 0, 255]; // red
      canvas.set(x, y, color);
    });

    return canvas.buffer;
  }
}

/** Traverse range of values for two parameters of the input set,
 * use mean input values for all other parameters,
 * generate heatmap for the values in range,
 * overlay values from output set,
 * and generate ANSI digram printable on console.
 */
export class Scatter {
  /** Row of mean values from all columns */
  private readonly means: Row;

  /** Smallest input value in column for x-axis */
  private readonly xmin: number;

  /** Largest input value in column for x-axis */
  private readonly xmax: number;

  /** Smallest input value in column for y-axis */
  private readonly ymin: number;

  /** Largest input value in column for y-axis */
  private readonly ymax: number;

  /**
   * @param   {Network}   network       Neural network
   * @param   {DataSet}   input         Training input values, array of rows of numbers
   * @param   {DataSet}   output        Training output values, array of rows of numbers
   * @param   {number}    xcol          Column number in input to use at x-axis
   * @param   {number}    ycol          Column number in input to use at y-axis
   * @param   {number}    zcol          Column number in output to use as value
   * @param   {string[]}  inputLabels   Column names in input
   * @param   {string[]}  outputLabels  Column names in input
   */
  constructor(
    private readonly network: Network,
    private readonly input: DataSet,
    private readonly output: DataSet,
    private readonly xcol: number,
    private readonly ycol: number,
    private readonly zcol: number,
    private readonly inputLabels: string[],
    private readonly outputLabels: string[]
  ) {
    const first: Row = input[0];
    this.means = Array(first.length);
    const meanIndex = Math.round(first.length);
    const lastIndex = input.length - 1;
    let xmin: number = 0,
      xmax: number = 1,
      ymin: number = 0,
      ymax: number = 1;
    first.forEach((_, colIndex) => {
      const column: Column = input.map((row) => row[colIndex]).sort();
      switch (colIndex) {
        case xcol:
          // Min and Max values for x-axis
          xmin = column[0];
          xmax = column[lastIndex];
          break;
        case ycol:
          // Min and Max values for y-axis
          ymin = column[0];
          ymax = column[lastIndex];
          break;
        default:
          // Populate input row of mean values
          this.means[colIndex] = column[meanIndex];
      }
    });
    this.xmin = xmin;
    this.xmax = xmax;
    this.ymin = xmin;
    this.ymax = xmax;
  }

  /** Return middle value of sorted values */
  private static mean(col: Column): number {
    const sorted = col.slice().sort();
    const middle: number = Math.round(sorted.length / 2);
    return sorted[middle];
  }

  /** Extract all values from one column */
  private static column(grid: DataSet, n: number): Column {
    return grid.map((row: Row) => row[n]);
  }

  /** Get grid of values from Network */
  private values(xcount: number, ycount: number): DataSet {
    const result: number[][] = [];
    for (
      let y = this.ymin;
      y <= this.ymax;
      y += (this.ymax - this.ymin) / (ycount - 1)
    ) {
      const row: Row = [];
      for (
        let x = this.xmin;
        x <= this.xmax;
        x += (this.xmax - this.xmin) / (xcount - 1)
      ) {
        const input = this.means;
        input[this.xcol] = x;
        input[this.ycol] = y;
        const output: Row = this.network.predict(input);
        const value: number = output[this.zcol];
        row.push(value);
      }
      result.push(row);
    }
    return result;
  }

  /** Generate the unannotated heatmap */
  private heatmap(width: number, height: number): string {
    return "";
  }

  /** Diagram Layout:
   *  ▯ Lowest output  ▮ Highest output
   *  Highest_Y┌────────────────────────┐
   *           │                        │
   * Y_Axisname│        heatmap         │
   *           │                        │
   *   Lowest_Y└────────────────────────┘
   *       Lowest_X  Y_Axisname  Lowest_Y
   */
  private layout(): number {
    // Reserved space for labels on Y axis
    // TODO: Calculate max of [Lowest_Y, Highest_Y and Y_Axisname]
    return 2;
  }

  /** Generate heatmap diagram
   * @param [width=40] Number of chars wide
   * @param [height=10] Number of chars high
   * @result String printable on terminal console
   */
  public plot(width = 40, height = 10): string {
    const area = width * height;
    const v = this.values(8, 4);
    console.log(v);
    return "" + area;
  }
}
