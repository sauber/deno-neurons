import { avg, std } from "@sauber/statistics";
import { Value } from "./value.ts";
import { Node } from "./node.ts";
import { Neuron, Normalizer } from "./neuron.ts";
import type { NeuronData, NormalizerData } from "./neuron.ts";
import type { Inputs } from "./train.ts";

/** An exported layer of NeuronData */
export type DenseData = Array<NeuronData>;

/** An exported layer of NormalizerData */
export type DeviationData = Array<NormalizerData>;

type Offset = {
  offset: number;
  factor: number;
};

/** Connect one input to one output */
export class Simple extends Node {
  constructor(
    public readonly inputs: number,
    private readonly neurons: Array<Neuron> = Array.from(
      Array(inputs),
      (_) => new Neuron(1)
    )
  ) {
    super();
  }

  public static import(data: DenseData): Simple {
    const neurons = data.map((data) => Neuron.import(data));
    return new Simple(neurons.length, neurons);
  }

  public override get export(): DenseData {
    return this.neurons.map((n) => n.export);
  }

  public forward(x: Value[]): Value[] {
    return this.neurons.map((n, i) => n.forward([x[i]]));
  }

  public override get parameters(): Value[] {
    const params: Value[] = [];
    for (const neuron of this.neurons) params.push(...neuron.parameters);
    return params;
  }
}

/** All nodes in input layer connected to all nodes in output layer */
export class Dense extends Node {
  constructor(
    public readonly inputs: number,
    public readonly outputs: number,
    private readonly neurons: Array<Neuron> = Array.from(
      Array(outputs),
      (_) => new Neuron(inputs)
    )
  ) {
    super();
  }

  public static import(data: DenseData): Dense {
    const neurons = data.map((data) => Neuron.import(data));
    const inputs = neurons[0]?.inputs || 0;
    const outputs = neurons.length;
    return new Dense(inputs, outputs, neurons);
  }

  public override get export(): DenseData {
    return this.neurons.map((n) => n.export);
  }

  public forward(x: Value[]): Value[] {
    return this.neurons.map((n) => n.forward(x));
  }

  public override get parameters(): Value[] {
    const params: Value[] = [];
    for (const neuron of this.neurons) params.push(...neuron.parameters);
    return params;
  }
}

/** Relu Activation Layer */
export class Relu extends Node {
  public forward(x: Value[]): Value[] {
    return x.map((n) => n.relu());
  }
}

/** Leaky Relu Activation Layer */
export class LRelu extends Node {
  public forward(x: Value[]): Value[] {
    return x.map((n) => n.lrelu());
  }
}

/** Sigmoid Activation Layer */
export class Sigmoid extends Node {
  public forward(x: Value[]): Value[] {
    return x.map((n) => n.sigmoid());
  }
}

/** Tanh Activation Layer */
export class Tanh extends Node {
  public forward(x: Value[]): Value[] {
    return x.map((n) => n.tanh());
  }
}

/** Shift and scale inputs into a distribution centered around 0 with standard deviation 1 */
export class Normalize extends Node {
  constructor(
    public readonly inputs: number,
    private readonly scalers: Array<Normalizer> = Array.from(
      Array(inputs),
      (_) => new Normalizer()
    )
  ) {
    super();
  }

  public forward(x: Value[]): Value[] {
    return this.scalers.map((scaler, index) => scaler.forward(x[index]));
  }

  /** Calculate means and variance of input, and set offset and factor accordingly */
  public override  adapt(inputs: Inputs): void {
    inputs[0].forEach((_, index) => {
      const col: number[] = inputs.map((r) => r[index]);
      const mean: number = avg(col);
      const variance: number = std(col);
      // console.log({ mean, variance });
      this.scalers[index] = new Normalizer(
        new Value(mean),
        new Value(variance)
      );
    });
  }

  /** Export layer of normalizers */
  public override get export(): DeviationData {
    return this.scalers.map((s) => s.export);
  }

  /** Generate layer of normalizers from expected mean and variance of input columns */
  public static import(data: DeviationData): Normalize {
    return new Normalize(
      data.length,
      data.map((n) => new Normalizer(new Value(n.mean), new Value(n.variance)))
    );
  }
}
