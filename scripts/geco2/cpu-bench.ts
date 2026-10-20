import { cpuBenchmark } from '../../src/ai/device';
console.log('cpuBenchmark ms', [cpuBenchmark(), cpuBenchmark(), cpuBenchmark()].map((v) => v.toFixed(1)).join(' '));
