export type Row = Array<number>;
export type Inputs = Array<Row>;
export type Outputs = Array<Row>;

/** Extract column from Inputs or Outputs */
function column(grid: Inputs | Outputs, n: number): Column {
  return new Column(grid.map((row: Row) => row[n]));
}

/** A column of numbers */
export class Column {
  private readonly sorted: number[];

  /** Sort numbers */
  constructor(public readonly data: number[]) {
    this.sorted = data.slice().sort();
  }

  /** Largest number */
  public get max(): number {
    return this.sorted[this.sorted.length - 1];
  }

  /** Smallest number */
  public get min(): number {
    return this.sorted[0];
  }

  /** Middle number */
  public get mean(): number {
    const index = Math.round((this.sorted.length-1) / 2);
    return this.sorted[index];
  }

  /** Evenly distributed points between min and max, both included */
  public points(count: number): number[] {
    const low: number = this.min;
    const high: number = this.max;
    return Array(count)
      .fill(0)
      .map((_, i) => low + (i * (high - low)) / (count - 1));
  }
}

/** Input and output training data */
export class Training {
  /** Values in X column */
  public readonly xs: Column;

  /** Values in Y column */
  public readonly ys: Column;

  /** Output Values */
  public readonly values: Column;

  constructor(
    /** Input training data */
    private readonly input: Inputs,
    /** Output training data */
    private readonly output: Outputs,
    /** Column number in input for x-axis */
    private readonly xcol: number,
    /** Column number in input for y-axis */
    private readonly ycol: number,
    /** Column number in output for z-axis */
    public readonly zcol: number
  ) {
    this.xs = column(this.input, this.xcol);
    this.ys = column(this.input, this.ycol);
    this.values = column(this.output, this.zcol);
  }

  /** Mean value from each input column, except columns used for X and Y axis */
  private get means(): Row {
    const first: Row = this.input[0];
    const means = Array(first.length);
    // Indices of columns not used in heat map
    const columns: number[] = first
      .map((_, i) => i)
      .filter((i) => i != this.xcol && i != this.ycol);
    columns.forEach((i) => {
      const sorted: Column = column(this.input, i);
      means[i] = sorted.mean;
    });
    return means;
  }

  /**
   * Generate a new set of inputs covering X and Y ranges.
   * Use mean values in all columns except X and Y.
   */
  public scatter(xcount: number, ycount: number): Inputs {
    const result: Inputs = [];
    const xs: number[] = this.xs.points(xcount);
    const ys: number[] = this.ys.points(ycount);
    const means = this.means;
    ys.forEach((y) => {
      xs.forEach((x) => {
        const row: Row = [...means];
        row[this.xcol] = x;
        row[this.ycol] = y;
        result.push(row);
      });
    });
    return result;
  }
}
