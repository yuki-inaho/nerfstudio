#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

function parseArgs(argv) {
  const args = {
    spzVersion: 4,
    modulePath: null
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--input') {
      args.input = argv[++i];
    } else if (arg === '--output') {
      args.output = argv[++i];
    } else if (arg === '--spz-version') {
      args.spzVersion = Number(argv[++i]);
    } else if (arg === '--module-path') {
      args.modulePath = argv[++i];
    } else if (arg === '-h' || arg === '--help') {
      args.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function usage() {
  console.log(`Usage: node scripts/convert_splat_to_spz.mjs --input splat.ply --output splat.spz

Options:
  --spz-version 3|4       SPZ format version to write. Default: 4
  --module-path PATH      Optional path to @playcanvas/splat-transform/dist/index.mjs
`);
}

function formatDuration(ms) {
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  }
  const totalSeconds = ms / 1000;
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(2)}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}m ${seconds.toFixed(1)}s`;
}

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

async function timedStage(name, callback) {
  const start = performance.now();
  console.log(`[${new Date().toISOString()}] ${name} started`);
  const heartbeat = setInterval(() => {
    console.log(`[${new Date().toISOString()}] ${name} still running (${formatDuration(performance.now() - start)})`);
  }, 10000);
  try {
    const result = await callback();
    clearInterval(heartbeat);
    console.log(`[${new Date().toISOString()}] ${name} finished in ${formatDuration(performance.now() - start)}`);
    return result;
  } catch (error) {
    clearInterval(heartbeat);
    console.error(`[${new Date().toISOString()}] ${name} failed after ${formatDuration(performance.now() - start)}`);
    throw error;
  }
}

async function importSplatTransform(modulePath) {
  if (modulePath) {
    return import(pathToFileURL(path.resolve(modulePath)).href);
  }

  try {
    return await import('@playcanvas/splat-transform');
  } catch {
    const npmRoot = execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
    const globalModule = path.join(npmRoot, '@playcanvas', 'splat-transform', 'dist', 'index.mjs');
    return import(pathToFileURL(globalModule).href);
  }
}

async function main() {
  const totalStart = performance.now();
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  if (!args.input || !args.output) {
    usage();
    process.exitCode = 2;
    return;
  }
  if (![3, 4].includes(args.spzVersion)) {
    throw new Error(`Unsupported SPZ version: ${args.spzVersion}`);
  }

  const inputPath = path.resolve(args.input);
  const outputPath = path.resolve(args.output);
  const inputName = path.basename(inputPath);
  const outputName = path.basename(outputPath);

  console.log(`[${new Date().toISOString()}] SPZ conversion started`);
  const st = await timedStage('load @playcanvas/splat-transform', () => importSplatTransform(args.modulePath));
  const inputBytes = await timedStage('read input PLY', async () => new Uint8Array(await readFile(inputPath)));
  console.log(`Input: ${inputPath} (${formatBytes(inputBytes.byteLength)})`);
  const readFs = new st.MemoryReadFileSystem();
  readFs.set(inputName, inputBytes);

  const options = {
    iterations: 10,
    lodSelect: [],
    unbundled: false,
    lodChunkCount: 512,
    lodChunkExtent: 16,
    spzVersion: args.spzVersion
  };

  const tables = await timedStage('decode input PLY', () => st.readFile({
    filename: inputName,
    inputFormat: st.getInputFormat(inputName),
    options,
    params: [],
    fileSystem: readFs
  }));
  if (tables.length !== 1) {
    throw new Error(`Expected one DataTable, got ${tables.length}`);
  }

  const writeFs = new st.MemoryFileSystem();
  await timedStage('encode SPZ', () => st.writeFile({
    filename: outputName,
    outputFormat: st.getOutputFormat(outputName, options),
    dataTable: tables[0],
    options
  }, writeFs));

  const outputBytes = writeFs.results.get(outputName);
  if (!outputBytes) {
    throw new Error(`splat-transform did not produce ${outputName}`);
  }

  await timedStage('write output SPZ', () => writeFile(outputPath, outputBytes));
  console.log(`splat-transform ${st.version ?? 'unknown'}`);
  console.log(`SPZ v${args.spzVersion}: ${outputPath} (${outputBytes.byteLength} bytes, ${formatBytes(outputBytes.byteLength)})`);
  console.log(`[${new Date().toISOString()}] SPZ conversion finished in ${formatDuration(performance.now() - totalStart)}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
