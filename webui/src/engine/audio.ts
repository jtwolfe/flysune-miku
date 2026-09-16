import { SAMPLE_RATE } from "./speaker";

export function floatToWav(samples: Float32Array, sampleRate = SAMPLE_RATE): Blob {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]!));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

let ctx: AudioContext | null = null;

export function getAudioContext(): AudioContext {
  if (!ctx) ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  return ctx;
}

export async function playSamples(
  samples: Float32Array,
  sampleRate = SAMPLE_RATE,
  offsetSamples = 0,
): Promise<AudioBufferSourceNode> {
  const ac = getAudioContext();
  if (ac.state === "suspended") await ac.resume();
  const start = Math.max(0, Math.min(samples.length, Math.floor(offsetSamples)));
  const slice = samples.subarray(start);
  const buffer = ac.createBuffer(1, Math.max(1, slice.length), sampleRate);
  buffer.copyToChannel(new Float32Array(slice), 0);
  const src = ac.createBufferSource();
  src.buffer = buffer;
  src.connect(ac.destination);
  src.start();
  return src;
}

export function waitForSource(src: AudioBufferSourceNode): Promise<void> {
  return new Promise((resolve) => {
    src.onended = () => resolve();
  });
}

export function downsamplePeaks(samples: Float32Array, buckets: number): Float32Array {
  const peaks = new Float32Array(buckets);
  if (!samples.length) return peaks;
  const step = samples.length / buckets;
  for (let i = 0; i < buckets; i++) {
    const a = Math.floor(i * step);
    const b = Math.min(samples.length, Math.floor((i + 1) * step));
    let m = 0;
    for (let j = a; j < b; j++) m = Math.max(m, Math.abs(samples[j]!));
    peaks[i] = m;
  }
  return peaks;
}
