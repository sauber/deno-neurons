import { sum, Value } from "./value.ts";
import { Node } from "./node.ts";

export type NeuronData = {
  bias: number;
  weights: Array<number>;
};

export type NormalizerData = {
  mean: number;
  variance: number;
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
  public override  get export(): NeuronData {
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

  public override get parameters(): Value[] {
    return [...this.weights, this.bias];
  }
}

/** Rescale input to mean 0 and standard deviation 1 */
export class Normalizer extends Node {
  constructor(
    private readonly mean: Value = randomValue(),
    private readonly variance: Value = randomValue()
  ) {
    super();
  }

  /** Add bias before factoring */
  public forward(input: Value): Value {
    // return input.add(this.bias).mul(this.weight);
    return input.sub(this.mean).div(this.variance);
  }

  /** Re-initialize a pre-trained neuron */
  public static import(data: NormalizerData): Normalizer {
    return new Normalizer(new Value(data.mean), new Value(data.variance));
  }

  /** Export bias and weights */
  public override get export(): NormalizerData {
    return { mean: this.mean.data, variance: this.variance.data };
  }
}
