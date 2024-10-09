import { Value } from "./value.ts";
import { Node } from "./node.ts";
import { Neuron, Scaler, Rescaler } from "./neuron.ts";
import type { NeuronData, ScalerData } from "./neuron.ts";

export type DenseData = Array<NeuronData>;
export type SimpleData = Array<ScalerData>;

type Offset = {
  offset: number;
  factor: number;
};
export type ScaleData = Array<Offset>;

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

  /** Forward propagation of value */
  public forward(x: Value[]): Value[] {
    return this.rescalers.map((rescaler, index) => rescaler.forward(x[index]));
  }

  /** Export array of Rescaler data */
  public get export(): SimpleData {
    return this.rescalers.map((r) => r.export);
  }

  /** Import array of Rescaler data */
  public static import(data: SimpleData): Rescale {
    const rescalers: Array<Rescaler> = data.map((data: ScalerData) =>
      Rescaler.import(data)
    );
    return new Rescale(rescalers.length, rescalers);
  }

  public get parameters(): Value[] {
    const params: Value[] = [];
    for (const rescaler of this.rescalers) params.push(...rescaler.parameters);
    return params;
  }
}

/** Scale input data before entering layers of neurons */
export class Scale extends Node {
  constructor(private readonly scales: ScaleData) {
    super();
  }

  public forward(x: Value[]): Value[] {
    return this.scales.map(
      (scale, index) => new Value(x[index].data * scale.factor + scale.offset)
    );
  }

  public get export(): ScaleData {
    return this.scales;
  }

  public static import(data: ScaleData): Scale {
    return new Scale(data);
  }
}
