deno  run --v8-flags=--prof .\example.ts
node --prof-process .\isolate-*-v8.log > prof.log
rm .\isolate-*-v8.log