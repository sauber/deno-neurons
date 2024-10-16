import { sum, Value } from "./value.ts";
import type { Network } from "./network.ts";

/** Array of rows of input data */
export type Inputs = number[][];

/** Array of rows of output data */
export type Outputs = number[][];

/** Input or output data type casted to Value objects */
export type Values = Value[][];

/** Calculate mean square error for a batch of actual vs expected outputs */
export function MeanSquareError(a: Values, b: Values): Value {
  const squares: Value[] = a
    .map((line: Value[], row: number) =>
      line.map((val: Value, col: number) => val.sub(b[row][col]).pow(2))
    )
    .flat();
  const count: Value = new Value(a.length);
  const mean: Value = sum(...squares).div(count);
  return mean;
}

/** Train a neural network */
export class Train {
  private readonly lossHistory: number[] = [];
  private readonly xs: Values;
  private readonly ys: Values;

  // Stop when loss is lower than epsilon
  public epsilon = 0.0001;

  // Number of samples per step
  public batchSize = 32;

  /** Callback funtion */
  public callback: (iteration: number, loss: number[]) => void = () => {};

  /** Number of iterations between callbacks */
  public callbackFrequency: number = 100;

  constructor(
    public readonly network: Network,
    public readonly inputs: Inputs,
    public readonly outputs: Outputs
  ) {
    this.xs = inputs.map((row) => row.map((v) => new Value(v)));
    this.ys = outputs.map((row) => row.map((v) => new Value(v)));
  }

  /** Pick random samples for training */
  private batch(): [Values, Values] {
    const xs: Values = [];
    const ys: Values = [];
    const l = this.xs.length;
    for (let n = 0; n < this.batchSize; ++n) {
      const i = Math.floor(Math.random() * l);
      xs.push(this.xs[i]);
      ys.push(this.ys[i]);
    }
    return [xs, ys];
  }

  /** Run training on a batch */
  private step(learning_rate: number): void {
    const [xs, ys] = this.batch();

    // Forward
    const predict: Values = xs.map((line: Value[]) =>
      this.network.forward(line)
    );
    const loss = MeanSquareError(ys, predict);
    if (isNaN(loss.data) || loss.data > 1000000) {
      // loss.print();
      throw new Error(`Loss inclined to infinity: ${loss.data}`);
    }
    this.lossHistory.push(loss.data);

    // Backward
    this.network.zeroGrad();
    loss.backward();

    // Update
    // Stochastic Gradient Descent
    for (const p of this.network.parameters) {
      p.data -= learning_rate * p.grad;
      if (!isFinite(p.data) || Math.abs(p.data) > 1000000) {
        // loss.print();
        p.print();
        console.log({ learning_rate, grad: p.grad });
        throw new Error("Data approaching Infinity");
      }
    }
  }

  /** Iterate until loss is less than epsilon or until max iterations reached */
  public run(iterations: number = 1000, rate: number = 0.1): number {
    let i = 1;
    for (; i <= iterations; i++) {
      this.step(rate);
      if (i % this.callbackFrequency == 0 && i > 0)
        this.callback(i, this.lossHistory);
      const l = this.lossHistory.length;
      // Stop when loss is small enough
      if (this.lossHistory[this.lossHistory.length - 1] < this.epsilon) break;
      // Stop when loss is unchanged
      if (l >= 2 && this.lossHistory[l - 1] == this.lossHistory[l - 2]) break;
      // eta.sync_update(i);
    }
    --i;
    if (i % this.callbackFrequency != 0) this.callback(i, this.lossHistory);
    return i;
  }
}
