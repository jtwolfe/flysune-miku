import { encodeEnhancedContext } from "./encode";
import { PHONEME_LIST, type Phoneme, type PickerManifest, type Vote } from "./types";

export type PickerResult = {
  winner: Phoneme;
  confidence: number;
  votes: Vote[];
  nKcActive: number;
  kcActive: number[];
};

function viewArray(buf: ArrayBuffer, spec: { offset: number; shape: number[]; nbytes: number }) {
  return new Float32Array(buf, spec.offset, spec.nbytes / 4);
}

/** vec (rows) @ matrix (rows, cols) -> (cols) */
function matvec(matrix: Float32Array, rows: number, cols: number, vec: Float32Array): Float32Array {
  const out = new Float32Array(cols);
  for (let j = 0; j < cols; j++) {
    let s = 0;
    for (let i = 0; i < rows; i++) {
      s += vec[i]! * matrix[i * cols + j]!;
    }
    out[j] = s;
  }
  return out;
}

export class MoreFlyPicker {
  readonly manifest: PickerManifest;
  private inputToPn: Float32Array;
  private pnKc: Float32Array;
  private weights: Record<string, Float32Array>;

  constructor(manifest: PickerManifest, bin: ArrayBuffer) {
    this.manifest = manifest;
    const a = manifest.arrays;
    this.inputToPn = viewArray(bin, a.input_to_pn!);
    this.pnKc = viewArray(bin, a.pn_kc_weights!);
    this.weights = {};
    for (const p of PHONEME_LIST) {
      this.weights[p] = viewArray(bin, a[`weights_${p}`]!);
    }
  }

  static async load(base = "/models"): Promise<MoreFlyPicker> {
    const [manRes, binRes] = await Promise.all([
      fetch(`${base}/picker.json`),
      fetch(`${base}/picker.bin`),
    ]);
    if (!manRes.ok || !binRes.ok) throw new Error("Failed to load picker model");
    const manifest = (await manRes.json()) as PickerManifest;
    const bin = await binRes.arrayBuffer();
    return new MoreFlyPicker(manifest, bin);
  }

  encodeToKc(
    letterContext: string,
    phonemePos: number | null,
    nPhonemes: number | null,
    previousPhone: Phoneme | null,
  ): { kc: Float32Array; nActive: number; active: number[] } {
    const { inputDim, nPn, nKc, kcSparsity, contextSize } = this.manifest;
    const feats = encodeEnhancedContext(
      letterContext,
      phonemePos,
      nPhonemes,
      previousPhone,
      contextSize,
    );
    if (feats.length !== inputDim) {
      throw new Error(`Feature dim ${feats.length} != ${inputDim}`);
    }
    const pnRaw = matvec(this.inputToPn, inputDim, nPn, feats);
    let max = 0;
    for (let i = 0; i < nPn; i++) {
      const v = pnRaw[i]! < 0 ? 0 : pnRaw[i]!;
      pnRaw[i] = v;
      if (v > max) max = v;
    }
    if (max > 0) {
      for (let i = 0; i < nPn; i++) pnRaw[i] = pnRaw[i]! / max;
    }
    const kcInput = matvec(this.pnKc, nPn, nKc, pnRaw);
    const nActive = Math.max(1, Math.trunc(nKc * kcSparsity));
    const sorted = Array.from(kcInput);
    sorted.sort((a, b) => a - b);
    const threshold = sorted[nKc - nActive]!;
    const kc = new Float32Array(nKc);
    const active: number[] = [];
    for (let i = 0; i < nKc; i++) {
      if (kcInput[i]! >= threshold) {
        kc[i] = 1;
        active.push(i);
      }
    }
    return { kc, nActive: active.length, active };
  }

  predict(
    letterContext: string,
    phonemePos: number | null,
    nPhonemes: number | null,
    previousPhone: Phoneme | null,
  ): PickerResult {
    const { nKc } = this.manifest;
    const { kc, nActive, active } = this.encodeToKc(
      letterContext,
      phonemePos,
      nPhonemes,
      previousPhone,
    );
    const votes: Vote[] = [];
    for (const phoneme of PHONEME_LIST) {
      const w = this.weights[phoneme]!;
      let yes = 0;
      let no = 0;
      for (let i = 0; i < nKc; i++) {
        if (kc[i] === 0) continue;
        yes += kc[i]! * w[i * 2]!;
        no += kc[i]! * w[i * 2 + 1]!;
      }
      const isYes = yes < no;
      const margin = Math.abs(no - yes);
      const baseline = Math.abs(yes) + Math.abs(no) + 1e-6;
      const confidence = margin / baseline;
      votes.push({
        phoneme,
        isYes,
        confidence,
        score: isYes ? confidence : -confidence,
        mbonYes: yes,
        mbonNo: no,
      });
    }
    const { winner, confidence } = this.voteSoftmax(votes);
    votes.sort((a, b) => b.score - a.score);
    return { winner, confidence, votes, nKcActive: nActive, kcActive: active };
  }

  private voteSoftmax(votes: Vote[]): { winner: Phoneme; confidence: number } {
    const temp = Math.max(this.manifest.voteTemperature, 0.01);
    let maxS = -Infinity;
    for (const v of votes) maxS = Math.max(maxS, v.score / temp);
    let sum = 0;
    const exp: number[] = [];
    for (const v of votes) {
      const e = Math.exp(v.score / temp - maxS);
      exp.push(e);
      sum += e;
    }
    let bestI = 0;
    let bestP = -1;
    for (let i = 0; i < votes.length; i++) {
      const p = exp[i]! / sum;
      if (p > bestP) {
        bestP = p;
        bestI = i;
      }
    }
    return { winner: votes[bestI]!.phoneme, confidence: bestP };
  }
}
