import { BarLine } from "./dashboard/barline.ts";
import { Loss } from "./dashboard/loss.ts";
import { Scatter } from "./dashboard/scatter.ts";
import { Iteration } from "./dashboard/iteration.ts";
import type { Train } from "./train.ts";

const SEP = "│";
const ESC = "\u001B[";
const SHOW = ESC + "?25h";
const HIDE = ESC + "?25l";

export class Dashboard {
  private readonly scatter: Scatter;
  private readonly loss: Loss;
  private readonly header: string;
  private iteration: Iteration | undefined;

  constructor(
    private readonly train: Train,
    private readonly width: number = 79,
    private readonly height: number = 22
  ) {
    const colWidth = Math.floor((this.width - 1) / 2);
    const colHeight = this.height - 2;
    this.scatter = new Scatter(train.network, train.inputs, train.outputs, {
      width: colWidth,
      height: colHeight,
    });
    this.loss = new Loss(colWidth, colHeight);
    this.header = ["Scatter Plot", "Loss History"]
      .map((h) => new BarLine(colWidth, "-").left(h).line)
      .join(SEP);
  }

  // Combine components into dashboard
  public render(iteration: number, losses: number[]): string {
    const scatter: string[] = this.scatter.plot().split("\n");
    const loss: string[] = this.loss.render(losses).split("\n");
    return [
      this.header,
      ...scatter.map((line, index) => [line, loss[index]].join(SEP)),
      this.iteration?.render(iteration)
    ].join("\n");
  }

  /** Run iterations until network converged. Display dashboard at each N iteration */
  public run(
    iterations: number,
    learning_rate: number,
    frequency: number
  ): number {
    this.iteration = new Iteration(iterations, this.width);

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
