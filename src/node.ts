import type { Value } from "./value.ts";

export class Node {
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
}
