# deno-neurons

Pure-deno implementation of neural network.

## Example

Neural network to recognize a circular wave.

```ts
import { Network, Train, Dashboard } from "jsr:@sauber/neurons";

const network = new Network(2).dense(8).lrelu.dense(6).lrelu.dense(1).tanh;
const inputs: number[][] = [];
const outputs: number[][] = [];
for (let i = 0; i < 100; ++i) {
  const [x, y] = [Math.random() * 14 - 7, Math.random() * 14 - 7];
  inputs.push([x, y]);
  outputs.push([-Math.sin(Math.sqrt(x * x + y * y))]);
}

const train = new Train(network, inputs, outputs);
train.epsilon = 0.01;
const dashboard = new Dashboard(train, 78, 16);
const max_iteration = 20000;
const learning_rate = 0.2;
const count = train.run(max_iteration, learning_rate);
console.log("Network trained in", count, "iterations.");
```

## See Also

Use [ML-CLI-Dashboard](https://github.com/sauber/deno-dashboard) for ascii dashboard visualization during training.

## Credits

Based on [MicrogradTS](https://github.com/trekhleb/micrograd-ts)

## License

See [LICENSE](./LICENSE).