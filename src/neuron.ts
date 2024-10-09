import { sum, Value } from "./value.ts";
import { Node } from "./node.ts";

export type NeuronData = {
  bias: number;
  weights: Array<number>;
};

export type ScalerData = {
  max: number;
};

/** Random Value between -1 and +1 */
function randomValue(): Value {
  return new Value(Math.random() * 2 - 1, { op: "🔀" });
}

/** Neuron node with multiple weighted inputs and bias */
export class Neuron extends Node {
  constructor(
    public readonly inputs: number,
    private readonly bias: Value = randomValue(),
    private readonly weights: Array<Value> = Array.from(Array(inputs), (_) =>
      randomValue()
    )
  ) {
    super();
  }

  /** Re-initialize a pre-trained neuron */
  public static import(data: NeuronData): Neuron {
    const bias = new Value(data.bias);
    const weights = data.weights.map((v) => new Value(v));
    return new Neuron(weights.length, bias, weights);
  }

  /** Export bias and weights */
  public get export(): NeuronData {
    return {
      bias: this.bias.data,
      weights: this.weights.map((v: Value) => v.data),
    };
  }

  /** Calculate w·x + b */
  public forward(inputs: Value[]): Value {
    if (inputs.length != this.weights.length) {
      throw new Error(
        `Wrong number of input. Got ${inputs.length}, Expected ${this.weights.length}.`
      );
    }
    return sum(
      ...inputs.map((input: Value, index: number) =>
        input.mul(this.weights[index])
      ),
      this.bias
    );
  }

  public get parameters(): Value[] {
    return [...this.weights, this.bias];
  }
}

/** Node scaling input to -1:1 output range */
export class Scaler extends Node {
  private min = 0;
  private max = 0;
  private a = 0;
  private b = 0;

  /** Adjust scaling for each input
   * TODO: Freeze range when not in training mode
   */
  public forward(v: Value): Value {
    // Extend range
    if (v.data > this.max) {
      this.max = v.data;
      this.b = 1;
    }
    if (v.data < this.min) {
      this.min = v.data;
      this.a = -1;
    }

    // [0:0] range and 0 input value
    if (this.min == 0 && this.max == 0) return v;

    // Scale input value to a:b
    // Scaling formula: (b-a) * (v-min) / (max-min) + a
    return new Value(this.b - this.a)
      .mul(v.sub(this.min))
      .div(this.max - this.min)
      .add(this.a);
  }

  /** Parameters
   * TODO: Adjust parameters during back propagation instead of forward propagation
   */
  public get parameters(): Value[] {
    return [];
  }
}

/** Neuron node with multiple weighted inputs and bias */
export class Rescaler extends Node {
  private readonly cache = { data: 0 };
  private readonly factor: Value;

  constructor(private readonly max: number = 1) {
    super();
    this.cache.data = max;
    this.factor = new Value(max);
  }

  /** Re-initialize a pre-trained neuron */
  public static import(data: ScalerData): Rescaler {
    return new Rescaler(data.max);
  }

  /** Export bias and weights */
  public get export(): ScalerData {
    return { max: this.max };
  }

  /** Add bias before factoring */
  public forward(input: Value): Value {
    // return input.add(this.bias).mul(this.weight);
    return input.rescale(this.factor, this.cache);
  }

  public get parameters(): Value[] {
    return [this.factor];
  }
}
