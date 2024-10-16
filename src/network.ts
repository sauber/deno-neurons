import { Value } from "./value.ts";
import {
  Dense,
  Simple,
  Relu,
  LRelu,
  Tanh,
  Sigmoid,
  Normalize,
} from "./layer.ts";
import type { DenseData, DeviationData } from "./layer.ts";
import { Node } from "./node.ts";
import { Inputs } from "./train.ts";

type ActivationLayer = "Relu" | "LRelu" | "Sigmoid" | "Tanh";
type DenseLayer = { Dense: DenseData };
type SimpleLayer = { Simple: DenseData };
type NormalizeLayer = { Normalize: DeviationData };
type NeuronLayer = DenseLayer | SimpleLayer;

type Layer = Dense | Relu | LRelu | Sigmoid | Tanh | Simple;
type Layers = Array<Layer>;

type LayerData = ActivationLayer | NeuronLayer | NormalizeLayer;

/** Network parameters exported in JSON format */
export type NetworkData = {
  inputs: number;
  layers: Array<LayerData>;
};

/** Sequence of layers and function to predict output */
export class Network extends Node {
  constructor(
    private readonly inputs: number,
    private readonly layers: Layers = []
  ) {
    super();
  }

  public static import(data: NetworkData): Network {
    // let inputs: number = data.inputs;
    const layers: Layers = [];
    data.layers.forEach((layer) => {
      if (typeof layer === "string") {
        switch (layer) {
          case "Relu":
            layers.push(new Relu());
            break;
          case "LRelu":
            layers.push(new LRelu());
            break;
          case "Sigmoid":
            layers.push(new Sigmoid());
            break;
          case "Tanh":
            layers.push(new Tanh());
            break;
        }
      } else {
        Object.entries(layer).forEach(([type, data]) => {
          switch (type) {
            case "Dense":
              layers.push(Dense.import(data));
              break;
            case "Simple":
              layers.push(Simple.import(data));
              break;
            case "Normalize":
              layers.push(Normalize.import(data));
              break;
          }
        });
      }
    });

    return new Network(data.inputs, layers);
  }

  public override get export(): NetworkData {
    const layers: Array<LayerData> = [];
    this.layers.forEach((layer: Layer) => {
      const type: string = layer.constructor.name;
      switch (type) {
        case "Dense":
          layers.push({ Dense: layer.export as DenseData });
          break;
        case "Simple":
          layers.push({ Simple: layer.export as DenseData });
          break;
        case "Normalize":
          layers.push({ Normalize: layer.export as DeviationData });
          break;
        case "LRelu":
        case "Relu":
        case "Sigmoid":
        case "Tanh":
          layers.push(type);
          break;
      }
    });

    return { inputs: this.inputs, layers };
  }

  public override get parameters(): Value[] {
    const params: Value[] = [];
    this.layers.forEach((layer: Layer) => params.push(...layer.parameters));
    return params;
  }

  /** Run a set of values through forward propagation and record the output */
  public predict(x: number[]): number[] {
    const xs = x.map((n) => new Value(n));
    const ys = this.forward(xs);
    return ys.map((y) => y.data);
  }

  public forward(xin: Value[]): Value[] {
    let xout = [...xin];
    for (const layer of this.layers) xout = layer.forward(xout);
    return xout;
  }

  private get outputs(): number {
    // Search for last layer where outputs is defined
    for (let i = this.layers.length - 1; i >= 0; --i) {
      const layer = this.layers[i];
      if ("outputs" in layer) return layer.outputs;
    }
    return this.inputs;
  }

  /** Create Network with additional Layer */
  private add(layer: Layer): Network {
    return new Network(this.inputs, [...this.layers, layer]);
  }

  public dense(outputs: number): Network {
    const inputs: number = this.outputs;
    return this.add(new Dense(inputs, outputs));
  }

  public get simple(): Network {
    const inputs: number = this.outputs;
    return this.add(new Simple(inputs));
  }

  public get relu(): Network {
    return this.add(new Relu());
  }

  public get lrelu(): Network {
    return this.add(new LRelu());
  }

  public get sigmoid(): Network {
    return this.add(new Sigmoid());
  }

  public get tanh(): Network {
    return this.add(new Tanh());
  }

  public get normalize(): Network {
    return this.add(new Normalize(this.inputs));
  }

  public override adapt(input: Inputs): void {
    this.layers.forEach((l) => l.adapt(input));
  }
}
