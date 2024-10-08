import type { Value } from "./value.ts";
import { Node } from "./node.ts";
import { Neuron, Scaler, Rescaler } from "./neuron.ts";
import type { NeuronData } from "./neuron.ts";

export type DenseData = Array<NeuronData>;

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

  public get export(): DenseData {
    return this.neurons.map((n) => n.export);
  }

  public forward(x: Value[]): Value[] {
    return this.neurons.map((n, i) => n.forward([x[i]]));
  }

  public get parameters(): Value[] {
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

  public get export(): DenseData {
    return this.neurons.map((n) => n.export);
  }

  public forward(x: Value[]): Value[] {
    return this.neurons.map((n) => n.forward(x));
  }

  public get parameters(): Value[] {
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

export class Normalize extends Node {
  constructor(
    public readonly inputs: number,
    private readonly scalers: Array<Scaler> = Array.from(
      Array(inputs),
      (_) => new Scaler()
    )
  ) {
    super();
  }

  public forward(x: Value[]): Value[] {
    return this.scalers.map((scaler, index) => scaler.forward(x[index]));
  }
}

export class Rescale extends Node {
  constructor(
    public readonly inputs: number,
    private readonly rescalers: Array<Rescaler> = Array.from(
      Array(inputs),
      (_) => new Rescaler()
    )
  ) {
    super();
  }

  public forward(x: Value[]): Value[] {
    return this.rescalers.map((rescaler, index) => rescaler.forward(x[index]));
  }
}
