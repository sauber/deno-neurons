import type { Network } from "../network.ts";
import type { Inputs, Row } from "./data.ts";

/** Pull predictions from neural network */
export class Prediction {
  constructor(
    private readonly network: Network,
    private readonly inputs: Inputs
  ) {}

  /** Get a single column of predictions */
  public outputs(zcol: number): number[] {
    return this.inputs.map(
      (input) => (this.network.predict(input) as Row)[zcol]
    );
  }
}
