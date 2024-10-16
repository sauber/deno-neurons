import type { Inputs } from "./train.ts";
import type { Value } from "./value.ts";

export abstract class Node {
  /** Reset gradient to 0 for all parameters */
  public zeroGrad() {
    this.parameters.forEach((p) => p.grad = 0);
  }

  /** List of prameters, default none */
  public get parameters(): Value[] {
    return [];
  }

  /** Export parameters, default none */
  public get export(): unknown {
    return {};
  }

  /** Preprocessing by adapting to input data */
  public adapt(_: Inputs): void {}
}
