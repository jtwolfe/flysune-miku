import {
  CONSONANTS,
  SEMIVOWELS,
  VOWELS,
} from "./phonemes";
import { PHONEME_LIST, PHONEME_TO_IDX, type Phoneme, type SpeakerParams } from "./types";

export const SAMPLE_RATE = 22050;
export const SPEAKER_DURATION_MIN_MS = 60;
export const SPEAKER_DURATION_MAX_MS = 250;

export const VOWEL_FORMANTS: Record<string, [number, number]> = {
  AA: [750, 1100], AE: [700, 1800], AH: [600, 1200], AO: [550, 850],
  EH: [550, 1900], ER: [500, 1400], IH: [400, 2000], IY: [280, 2400],
  UH: [450, 1050], UW: [310, 870],
  AW: [750, 1200], AY: [750, 1200], EY: [450, 2100], OW: [500, 850], OY: [550, 850],
};

export const DIPHTHONG_END: Record<string, [number, number]> = {
  AW: [350, 900], AY: [350, 2200], EY: [350, 2200], OW: [350, 900], OY: [350, 2200],
};

type Cons = {
  type: "stop" | "affricate" | "fricative" | "nasal" | "liquid";
  freq: number;
  noise?: number;
  voiced?: boolean;
  burst?: number;
  aspirate?: boolean;
  f2?: number;
  f3?: number;
};

export const CONSONANT_PARAMS: Record<string, Cons> = {
  P: { type: "stop", freq: 1800, noise: 0.7, voiced: false, burst: 0.5 },
  B: { type: "stop", freq: 300, noise: 0.4, voiced: true, burst: 0.3 },
  T: { type: "stop", freq: 3500, noise: 0.8, voiced: false, burst: 0.6 },
  D: { type: "stop", freq: 400, noise: 0.5, voiced: true, burst: 0.4 },
  K: { type: "stop", freq: 1500, noise: 0.8, voiced: false, burst: 0.5 },
  G: { type: "stop", freq: 350, noise: 0.5, voiced: true, burst: 0.3 },
  CH: { type: "affricate", freq: 4000, noise: 0.9, voiced: false },
  JH: { type: "affricate", freq: 2500, noise: 0.7, voiced: true },
  F: { type: "fricative", freq: 3500, noise: 0.6, voiced: false },
  V: { type: "fricative", freq: 250, noise: 0.4, voiced: true },
  TH: { type: "fricative", freq: 4500, noise: 0.4, voiced: false },
  DH: { type: "fricative", freq: 300, noise: 0.3, voiced: true },
  S: { type: "fricative", freq: 5500, noise: 1.0, voiced: false },
  Z: { type: "fricative", freq: 4500, noise: 0.8, voiced: true },
  SH: { type: "fricative", freq: 3500, noise: 0.9, voiced: false },
  ZH: { type: "fricative", freq: 2500, noise: 0.7, voiced: true },
  HH: { type: "fricative", freq: 1000, noise: 0.5, voiced: false, aspirate: true },
  M: { type: "nasal", freq: 250, f2: 1000, voiced: true },
  N: { type: "nasal", freq: 280, f2: 1700, voiced: true },
  NG: { type: "nasal", freq: 280, f2: 2300, voiced: true },
  L: { type: "liquid", freq: 350, f2: 1200, f3: 2800, voiced: true },
  R: { type: "liquid", freq: 350, f2: 1000, f3: 1600, voiced: true },
};

export const SEMIVOWEL_PARAMS: Record<string, { start: [number, number]; end: [number, number] }> = {
  W: { start: [350, 700], end: [400, 1200] },
  Y: { start: [300, 2300], end: [400, 1800] },
};

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number) {
  // Box-Muller
  const u = Math.max(1e-9, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function seedFor(phoneme: string, base: number) {
  let h = base;
  for (let i = 0; i < phoneme.length; i++) {
    h = Math.imul(h ^ phoneme.charCodeAt(i), 16777619);
  }
  return h >>> 0;
}

function linspace(start: number, end: number, n: number): Float32Array {
  const t = new Float32Array(n);
  if (n === 1) {
    t[0] = start;
    return t;
  }
  const step = (end - start) / (n - 1);
  for (let i = 0; i < n; i++) t[i] = start + step * i;
  return t;
}

function envelope(n: number, attack: number, release: number): Float32Array {
  const env = new Float32Array(n);
  env.fill(1);
  const atk = Math.min(Math.trunc(attack * SAMPLE_RATE), Math.floor(n / 3));
  const rel = Math.min(Math.trunc(release * SAMPLE_RATE), Math.floor(n / 3));
  for (let i = 0; i < atk; i++) env[i] = atk ? i / atk : 1;
  for (let i = 0; i < rel; i++) env[n - rel + i] = rel ? 1 - i / rel : 1;
  return env;
}

function gaussAmp(delta: number, width: number) {
  const t = delta / width;
  return Math.exp(-(t * t));
}

function mulEnv(signal: Float32Array, env: Float32Array) {
  const n = Math.min(signal.length, env.length);
  for (let i = 0; i < n; i++) signal[i] = signal[i]! * env[i]!;
}

function normalize(signal: Float32Array, peak: number) {
  let m = 0;
  for (let i = 0; i < signal.length; i++) m = Math.max(m, Math.abs(signal[i]!));
  if (m > 0) {
    const g = peak / m;
    for (let i = 0; i < signal.length; i++) signal[i] = signal[i]! * g;
  }
}

function parseParams(raw: Record<string, unknown>): SpeakerParams {
  return {
    f0: Number(raw.f0 ?? 140),
    f1Shift: Number(raw.f1_shift ?? 1),
    f2Shift: Number(raw.f2_shift ?? 1),
    f3Shift: Number(raw.f3_shift ?? 1),
    noiseLevel: Number(raw.noise_level ?? 0),
    durationMs: Number(raw.duration_ms ?? 100),
    attackMs: Number(raw.attack_ms ?? 10),
    releaseMs: Number(raw.release_ms ?? 15),
    harmonicRolloff: Number(raw.harmonic_rolloff ?? 0.7),
    prevPhoneF0Delta: Array.isArray(raw.prev_phone_f0_delta)
      ? (raw.prev_phone_f0_delta as number[])
      : undefined,
    prevPhoneDurDelta: Array.isArray(raw.prev_phone_dur_delta)
      ? (raw.prev_phone_dur_delta as number[])
      : undefined,
  };
}

export type SpeakerFile = {
  mode: string;
  seed: number;
  crossfadeMs: number;
  phonemes: string[];
  params: Record<string, Record<string, unknown>>;
};

export class SpeakerFly {
  phoneme: Phoneme;
  params: SpeakerParams;
  private rng: () => number;

  constructor(phoneme: Phoneme, params: SpeakerParams, seed: number) {
    this.phoneme = phoneme;
    this.params = params;
    this.rng = mulberry32(seedFor(phoneme, seed));
  }

  effective(prevPhone: Phoneme | null, position: number): SpeakerParams {
    const p = { ...this.params };
    if (prevPhone && p.prevPhoneF0Delta) {
      const i = PHONEME_TO_IDX[prevPhone];
      if (i != null) p.f0 += p.prevPhoneF0Delta[i] ?? 0;
    }
    if (prevPhone && p.prevPhoneDurDelta) {
      const i = PHONEME_TO_IDX[prevPhone];
      if (i != null) p.durationMs += p.prevPhoneDurDelta[i] ?? 0;
    }
    p.f0 *= 1.0 + 0.05 * (1.0 - position);
    p.durationMs = Math.max(SPEAKER_DURATION_MIN_MS, Math.min(SPEAKER_DURATION_MAX_MS, p.durationMs));
    return p;
  }

  synthesize(prevPhone: Phoneme | null, position: number): Float32Array {
    const params = this.effective(prevPhone, position);
    if (VOWELS.has(this.phoneme)) return this.vowel(params);
    if (SEMIVOWELS.has(this.phoneme)) return this.semivowel(params);
    if (CONSONANTS.has(this.phoneme)) return this.consonant(params);
    return this.fallback(params);
  }

  private vowel(params: SpeakerParams): Float32Array {
    const duration = params.durationMs / 1000;
    const n = Math.trunc(duration * SAMPLE_RATE);
    const t = linspace(0, duration, n);
    const base = VOWEL_FORMANTS[this.phoneme] ?? [500, 1500];
    let f1: Float32Array;
    let f2: Float32Array;
    if (this.phoneme in DIPHTHONG_END) {
      const end = DIPHTHONG_END[this.phoneme]!;
      f1 = linspace(base[0] * params.f1Shift, end[0] * params.f1Shift, n);
      f2 = linspace(base[1] * params.f2Shift, end[1] * params.f2Shift, n);
    } else {
      f1 = new Float32Array(n).fill(base[0] * params.f1Shift);
      f2 = new Float32Array(n).fill(base[1] * params.f2Shift);
    }
    const signal = new Float32Array(n);
    const f0 = params.f0;
    for (let h = 1; h < 12; h++) {
      const freq = f0 * h;
      for (let i = 0; i < n; i++) {
        const amp1 = gaussAmp(freq - f1[i]!, 150);
        const amp2 = gaussAmp(freq - f2[i]!, 200);
        const amp = (amp1 + amp2 * 0.6) / h ** params.harmonicRolloff;
        signal[i] += amp * Math.sin(2 * Math.PI * freq * t[i]!);
      }
    }
    mulEnv(signal, envelope(n, params.attackMs / 1000, params.releaseMs / 1000));
    if (params.noiseLevel > 0) {
      for (let i = 0; i < n; i++) signal[i] += gaussian(this.rng) * params.noiseLevel * 0.2;
    }
    normalize(signal, 0.75);
    return signal;
  }

  private consonant(params: SpeakerParams): Float32Array {
    const duration = params.durationMs / 1000;
    const n = Math.trunc(duration * SAMPLE_RATE);
    const t = linspace(0, duration, n);
    const cons = CONSONANT_PARAMS[this.phoneme] ?? {
      type: "fricative" as const,
      freq: 1000,
      noise: 0.5,
      voiced: false,
    };
    const freq = cons.freq * params.f1Shift;
    let signal: Float32Array;
    if (cons.type === "stop") signal = this.stop(n, t, freq, cons, params);
    else if (cons.type === "affricate") signal = this.affricate(n, t, freq, cons, params);
    else if (cons.type === "fricative") signal = this.fricative(n, t, freq, cons, params);
    else if (cons.type === "nasal") signal = this.nasal(n, t, freq, cons, params);
    else if (cons.type === "liquid") signal = this.liquid(n, t, freq, cons, params);
    else {
      signal = new Float32Array(n);
      for (let i = 0; i < n; i++) signal[i] = gaussian(this.rng) * 0.3;
    }
    normalize(signal, 0.65);
    return signal;
  }

  private stop(
    n: number,
    t: Float32Array,
    freq: number,
    cons: Cons,
    params: SpeakerParams,
  ): Float32Array {
    const signal = new Float32Array(n);
    let burstStart = Math.trunc(n * 0.35);
    let burstLen = Math.trunc(n * (cons.burst ?? 0.4));
    if (burstStart + burstLen > n) burstLen = n - burstStart;
    if (burstLen > 0) {
      for (let i = 0; i < burstLen; i++) {
        const noise = gaussian(this.rng) * (cons.noise ?? 0.5);
        const tBurst = i / SAMPLE_RATE;
        const carrier = Math.sin(2 * Math.PI * freq * tBurst);
        const env = Math.exp(-(i / burstLen) * 6);
        signal[burstStart + i] = (noise * 0.6 + carrier * 0.4 * Math.abs(noise)) * env;
      }
    }
    if (cons.voiced) {
      const env = envelope(n, 0.15, 0.15);
      for (let i = 0; i < n; i++) {
        signal[i] += Math.sin(2 * Math.PI * params.f0 * t[i]!) * 0.25 * env[i]!;
      }
    }
    return signal;
  }

  private affricate(
    n: number,
    t: Float32Array,
    freq: number,
    cons: Cons,
    params: SpeakerParams,
  ): Float32Array {
    const signal = new Float32Array(n);
    const stopLen = Math.trunc(n * 0.4);
    const tStop = linspace(0, stopLen / SAMPLE_RATE, stopLen);
    const stopPart = this.stop(stopLen, tStop, freq, {
      ...cons,
      noise: (cons.noise ?? 0.5) * 0.5,
      burst: 0.6,
    }, params);
    signal.set(stopPart.subarray(0, stopLen), 0);
    const fricLen = n - stopLen;
    if (fricLen > 0) {
      for (let i = 0; i < fricLen; i++) {
        const tt = i / SAMPLE_RATE;
        const noise = gaussian(this.rng) * (cons.noise ?? 0.5);
        const carrier = Math.sin(2 * Math.PI * freq * tt);
        signal[stopLen + i] = noise * 0.5 + carrier * 0.3 * noise;
      }
      const env = envelope(fricLen, 0.05, 0.1);
      for (let i = 0; i < fricLen; i++) signal[stopLen + i] *= env[i]!;
    }
    if (cons.voiced) {
      const env = envelope(n, 0.1, 0.1);
      for (let i = 0; i < n; i++) signal[i] += Math.sin(2 * Math.PI * params.f0 * t[i]!) * 0.2 * env[i]!;
    }
    return signal;
  }

  private fricative(
    n: number,
    t: Float32Array,
    freq: number,
    cons: Cons,
    params: SpeakerParams,
  ): Float32Array {
    const signal = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const noise = gaussian(this.rng) * ((cons.noise ?? 0.5) + params.noiseLevel) * 0.5;
      if (cons.aspirate) {
        signal[i] = noise * 0.5;
      } else {
        const carrier = Math.sin(2 * Math.PI * freq * t[i]!);
        signal[i] = noise * 0.4 + carrier * 0.2 * Math.abs(noise);
      }
    }
    mulEnv(
      signal,
      envelope(n, Math.max(0.05, params.attackMs / 1000), Math.max(0.08, params.releaseMs / 1000)),
    );
    if (cons.voiced) {
      const env = envelope(n, 0.1, 0.1);
      for (let i = 0; i < n; i++) signal[i] += Math.sin(2 * Math.PI * params.f0 * t[i]!) * 0.3 * env[i]!;
    }
    return signal;
  }

  private nasal(
    n: number,
    t: Float32Array,
    freq: number,
    cons: Cons,
    params: SpeakerParams,
  ): Float32Array {
    const f1 = freq * params.f1Shift;
    const f2 = (cons.f2 ?? 1000) * params.f2Shift;
    const signal = new Float32Array(n);
    for (let h = 1; h < 8; h++) {
      const hf = params.f0 * h;
      const amp1 = gaussAmp(hf - f1, 100);
      const amp2 = gaussAmp(hf - f2, 150) * 0.5;
      const amp = (amp1 + amp2) / h ** params.harmonicRolloff;
      for (let i = 0; i < n; i++) signal[i] += amp * Math.sin(2 * Math.PI * hf * t[i]!);
    }
    for (let i = 0; i < n; i++) signal[i] += Math.sin(2 * Math.PI * f1 * t[i]!) * 0.3;
    mulEnv(
      signal,
      envelope(n, Math.max(0.08, params.attackMs / 1000), Math.max(0.08, params.releaseMs / 1000)),
    );
    return signal;
  }

  private liquid(
    n: number,
    t: Float32Array,
    freq: number,
    cons: Cons,
    params: SpeakerParams,
  ): Float32Array {
    const f1 = freq * params.f1Shift;
    const f2 = (cons.f2 ?? 1200) * params.f2Shift;
    const f3 = (cons.f3 ?? 2800) * params.f3Shift;
    const signal = new Float32Array(n);
    for (let h = 1; h < 10; h++) {
      const hf = params.f0 * h;
      const amp1 = gaussAmp(hf - f1, 100);
      const amp2 = gaussAmp(hf - f2, 150) * 0.6;
      const amp3 = gaussAmp(hf - f3, 200) * 0.4;
      const amp = (amp1 + amp2 + amp3) / h ** params.harmonicRolloff;
      for (let i = 0; i < n; i++) signal[i] += amp * Math.sin(2 * Math.PI * hf * t[i]!);
    }
    mulEnv(
      signal,
      envelope(n, Math.max(0.08, params.attackMs / 1000), Math.max(0.1, params.releaseMs / 1000)),
    );
    return signal;
  }

  private semivowel(params: SpeakerParams): Float32Array {
    const duration = params.durationMs / 1000;
    const n = Math.trunc(duration * SAMPLE_RATE);
    const t = linspace(0, duration, n);
    const sv = SEMIVOWEL_PARAMS[this.phoneme] ?? { start: [350, 700], end: [400, 1200] };
    const f1 = linspace(sv.start[0] * params.f1Shift, sv.end[0] * params.f1Shift, n);
    const f2 = linspace(sv.start[1] * params.f2Shift, sv.end[1] * params.f2Shift, n);
    const signal = new Float32Array(n);
    for (let h = 1; h < 10; h++) {
      const freq = params.f0 * h;
      for (let i = 0; i < n; i++) {
        const amp1 = gaussAmp(freq - f1[i]!, 120);
        const amp2 = gaussAmp(freq - f2[i]!, 180) * 0.7;
        const amp = (amp1 + amp2) / h ** params.harmonicRolloff;
        signal[i] += amp * Math.sin(2 * Math.PI * freq * t[i]!);
      }
    }
    mulEnv(
      signal,
      envelope(n, Math.max(0.06, params.attackMs / 1000), Math.max(0.08, params.releaseMs / 1000)),
    );
    if (params.noiseLevel > 0) {
      for (let i = 0; i < n; i++) signal[i] += gaussian(this.rng) * params.noiseLevel * 0.15;
    }
    normalize(signal, 0.7);
    return signal;
  }

  private fallback(params: SpeakerParams): Float32Array {
    const duration = params.durationMs / 1000;
    const n = Math.trunc(duration * SAMPLE_RATE);
    const t = linspace(0, duration, n);
    const signal = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      signal[i] = Math.sin(2 * Math.PI * params.f0 * t[i]!) * 0.5 + gaussian(this.rng) * 0.1;
    }
    mulEnv(signal, envelope(n, 0.05, 0.05));
    return signal;
  }
}

export class SpeakerSwarmJs {
  flies: Record<string, SpeakerFly>;
  crossfadeMs: number;
  seed: number;

  constructor(file: SpeakerFile) {
    this.crossfadeMs = file.crossfadeMs;
    this.seed = file.seed;
    this.flies = {};
    for (const p of PHONEME_LIST) {
      const raw = file.params[p] ?? {};
      this.flies[p] = new SpeakerFly(p, parseParams(raw), file.seed);
    }
  }

  static async load(base = "/models"): Promise<SpeakerSwarmJs> {
    const res = await fetch(`${base}/speaker.json`);
    if (!res.ok) throw new Error("Failed to load speaker model");
    return new SpeakerSwarmJs((await res.json()) as SpeakerFile);
  }

  synthesizePhoneme(phoneme: Phoneme, prev: Phoneme | null, position: number): Float32Array {
    const fly = this.flies[phoneme];
    if (!fly) return new Float32Array(Math.trunc(0.08 * SAMPLE_RATE));
    return fly.synthesize(prev, position);
  }

  getParams(phoneme: Phoneme, prev: Phoneme | null, position: number): SpeakerParams {
    return this.flies[phoneme]!.effective(prev, position);
  }

  concatenate(phones: Phoneme[], crumbsOut?: Float32Array[]): Float32Array {
    if (!phones.length) return new Float32Array(0);
    const n = phones.length;
    const crumbs: Float32Array[] = [];
    crumbs.push(this.synthesizePhoneme(phones[0]!, null, n > 1 ? 0 : 0.5));
    for (let i = 1; i < n; i++) {
      const position = n > 1 ? i / (n - 1) : 0.5;
      crumbs.push(this.synthesizePhoneme(phones[i]!, phones[i - 1]!, position));
    }
    if (crumbsOut) crumbsOut.push(...crumbs.map((c) => c.slice()));
    return concatCrumbs(crumbs, this.crossfadeMs).audio;
  }
}

export function concatCrumbs(
  crumbs: Float32Array[],
  crossfadeMs: number,
): { audio: Float32Array; starts: number[] } {
  if (!crumbs.length) return { audio: new Float32Array(0), starts: [] };
  const xf = Math.trunc((crossfadeMs / 1000) * SAMPLE_RATE);
  const starts = [0];
  let result = crumbs[0]!.slice();
  for (let i = 1; i < crumbs.length; i++) {
    const audio = crumbs[i]!;
    if (xf > 0 && result.length >= xf && audio.length >= xf) {
      const start = result.length - xf;
      starts.push(start);
      const mixed = new Float32Array(result.length + audio.length - xf);
      mixed.set(result);
      for (let k = 0; k < xf; k++) {
        const fadeOut = 1 - k / (xf - 1 || 1);
        const fadeIn = k / (xf - 1 || 1);
        mixed[start + k] = result[start + k]! * fadeOut + audio[k]! * fadeIn;
      }
      mixed.set(audio.subarray(xf), result.length);
      result = mixed;
    } else {
      starts.push(result.length);
      const cat = new Float32Array(result.length + audio.length);
      cat.set(result);
      cat.set(audio, result.length);
      result = cat;
    }
  }
  return { audio: result, starts };
}
