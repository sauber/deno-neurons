import type { Network } from "./network.ts";
import { Training } from "./scatter/data.ts";
import type { Inputs, Outputs, Column, Row } from "./scatter/data.ts";
import { Prediction } from "./scatter/prediction.ts";
import { blockify } from "@image";

/** Generate a string of consecutive chars */
function pad(width: number, char: string = " "): string {
  return new Array(width).fill(char).join("");
}

type Overlay = [Column, Column, Column];

class HeatmapMaker {
  /** Row of mean values from all columns */
  private readonly prediction: Prediction;
  private readonly zcol: number;
  private readonly overlay: Overlay;

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

    this.overlay = [train.xs, train.ys, train.values];
  }

  public heatmap(): Heatmap {
    const values = this.prediction.outputs(this.zcol);
    const min: number = Math.min(...values, this.train.values.min);
    const max: number = Math.max(...values, this.train.values.max);
    return new Heatmap(values, this.overlay, min, max, this.width, this.height);
  }
}

class Heatmap {
  constructor(
    private readonly values: number[],
    private readonly overlay: Overlay,
    public readonly min: number,
    public readonly max: number,
    private readonly width: number,
    private readonly height: number
  ) {}

  /** Scale value to be in range 0-255 */
  private vscale(input: number): number {
    const input_min = this.min;
    const input_max = this.max;
    const output_min = 0;
    const output_max = 256;
    const f: number =
      ((input - input_min) / (input_max - input_min)) *
        (output_max - output_min) +
      output_min;

    const i: number = f > 255 ? 255 : Math.floor(f);
    return i;
  }

  /** Render heatmap as lines of strings */
  public render(): string[] {
    // const values = this.prediction.outputs(this.zcol);
    const ints: number[] = this.values.map((v) => this.vscale(v));
    const bitmap: Uint8Array = new Uint8Array(
      ints.map((i) => [i, i, i, 0]).flat()
    );

    // Overlay training 
    // console.log(this.overlay);
    const xmin = this.overlay[0].min;
    const xmax = this.overlay[0].max;
    const ymin = this.overlay[1].min;
    const ymax = this.overlay[1].max;
    const width = this.width * 2;
    const height = this.height * 2;
    this.overlay[0].data.forEach((x: number, i: number) => {
      // Scale X to xaxis point
      const xf = ((x - xmin) / (xmax - xmin)) * width;
      const xi = xf >= width ? width - 1 : Math.floor(xf);

      // Scale Y to yaxis point
      const y = this.overlay[1].data[i];
      const yf = ((y - ymin) / (ymax - ymin)) * height;
      const yi = yf >= height ? height - 1 : Math.floor(yf);

      // Scale v to output range
      const z = this.overlay[2].data[i];
      const v: number = this.vscale(z);

      // Calculate index in bitmap
      const index = (yi * width + xi) * 4;
      // console.log({ x,xmin,xmax, y, ymin,ymax, xf, yf, xi, yi, index });

      // Insert red or green pixel
      // if (index >= 0) 
        bitmap.set([255 - v, v, 64], index);
    });

    const printable: string = blockify(bitmap, this.width * 2, this.height * 2);
    const lines = printable.split("\n");
    return lines;
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
  private readonly maker: HeatmapMaker;

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

    const data = new Training(input, output, this.xcol, this.ycol, this.zcol);

    const xs: Column = data.xs;
    const ys: Column = data.ys;
    this.xaxis = new XAxis(this.xlabel, xs.max, xs.min, this.width);
    this.zaxis = new ZAxis(this.zlabel, this.width);
    const height: number = this.height - this.zaxis.height - this.xaxis.height;
    this.yaxis = new YAxis(this.ylabel, ys.max, ys.min, height);
    this.maker = new HeatmapMaker(
      this.width - this.yaxis.width,
      height,
      network,
      data
    );
  }

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

  /** Generate heatmap diagram
   * @param [width=40] Number of chars wide
   * @param [height=10] Number of chars high
   * @result String printable on terminal console
   */
  public plot(): string {
    const heatmap: Heatmap = this.maker.heatmap();
    const header = this.zaxis.render(heatmap.min, heatmap.max);
    const yaxis = this.yaxis.render();
    const content = heatmap.render();
    const footer = this.xaxis.render();

    const lines: string[] = [
      ...header,
      ...yaxis.map((line, index) => [line, content[index]].join("")),
      ...footer,
    ];

    return lines.join("\n");
  }
}
