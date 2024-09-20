import { plot } from "chart";
import { BarLine } from "./dashboard/barline.ts";
import { Scatter } from "./dashboard/scatter.ts";
import { Train } from "./train.ts";

const SEP = "│";
const ESC = "\u001B[";
const SHOW = ESC + "?25h";
const HIDE = ESC + "?25l";
// const UP = ESC + "14F";

/** Pick samples from array */
function samples(data: number[], count: number): number[] {
  if (count >= data.length) return data;
  const step = (data.length - 1) / count;
  const output: number[] = [];
  for (let i = 0; i < data.length; i += step) {
    output.push(data[Math.floor(i)]);
  }
  return output;
}

export class Dashboard {
  // Width of each column
  private readonly colWidth: number;
  private readonly colHeight: number;
  private readonly scatter: Scatter;

  constructor(
    private readonly train: Train,
    private readonly width: number = 79,
    private readonly height: number = 22
  ) {
    this.colWidth = Math.floor((this.width - 1) / 2);
    this.colHeight = this.height - 2;
    this.scatter = new Scatter(train.network, train.inputs, train.outputs, {
      width: this.colWidth,
      height: this.colHeight,
    });
  }

  /** Header across columns */
  private header(): string {
    return ["Scatter Plot", "Loss History"]
      .map((h) => new BarLine(this.colWidth, "-").left(h).line)
      .join(SEP);
  }

  /** Generate a chart of losses */
  private loss(losses: number[]): string {
    // Not enough data available
    if (losses.length < 2) {
      const blank = new BarLine(this.colWidth).line;
      const blanks = Array(this.colHeight).fill(blank);
      blanks[this.colHeight / 2] = new BarLine(this.colWidth).center(
        "No data"
      ).line;
      return blanks.join("\n");
    }

    // Generate a graph
    // TODO: Better estimation of padding width
    const loss: number[] = samples(losses, this.colWidth - 7);
    const printable = plot(loss, {
      height: this.colHeight - 1,
      padding: "     ",
    });
    return printable;
  }

  // Combine components for dashboard
  public render(iteration: number, losses: number[]): string {
    const scatter: string[] = this.scatter.plot().split("\n");
    const loss: string[] = this.loss(losses).split("\n");
    return [
      this.header(),
      ...scatter.map((line, index) => [line, loss[index]].join(SEP)),
      `Iteration: ${iteration}`,
    ].join("\n");
  }

  /** Run iterations until network converged. Display dashboard at each N iteration */
  public run(
    iterations: number,
    learning_rate: number,
    frequency: number
  ): number {
    let first: boolean = true;
    const lineup: string = ESC + "F";
    const home: string = ESC + (this.height + 1).toString() + "F";
    console.log(HIDE, lineup);

    // Define callback function
    this.train.callback = (iteration, losses) => {
      // Move cursor up to first line, except first time
      if (first) first = false;
      else console.log(home);

      console.log(this.render(iteration, losses));
    };
    this.train.callbackFrequency = frequency;
    const last: number = this.train.run(iterations, learning_rate);
    console.log(SHOW, lineup);
    return last;
  }
}
