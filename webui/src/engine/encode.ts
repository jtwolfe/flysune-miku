import { NUM_PHONEMES, PHONEME_TO_IDX, type Phoneme } from "./types";

export const CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz";
export const NUM_CHARS = CHAR_VOCAB.length;
const CHAR_TO_IDX: Record<string, number> = Object.fromEntries(
  [...CHAR_VOCAB].map((c, i) => [c, i]),
);

export function encodeEnhancedContext(
  letterContext: string,
  phonemePos: number | null,
  nPhonemes: number | null,
  previousPhone: Phoneme | null,
  contextSize = 3,
): Float32Array {
  const context = letterContext.toLowerCase();
  const contextLen = context.length;
  const parts: Float32Array[] = [];

  const posOnehot = new Float32Array(contextLen * NUM_CHARS);
  for (let i = 0; i < contextLen; i++) {
    const c = context[i]!;
    const idx = CHAR_TO_IDX[c] ?? CHAR_TO_IDX["_"]!;
    posOnehot[i * NUM_CHARS + idx] = 1;
  }
  parts.push(posOnehot);

  const bigram = new Float32Array(26 * 26);
  for (let i = 0; i < contextLen - 1; i++) {
    const c1 = context.charCodeAt(i);
    const c2 = context.charCodeAt(i + 1);
    if (c1 >= 97 && c1 <= 122 && c2 >= 97 && c2 <= 122) {
      bigram[(c1 - 97) * 26 + (c2 - 97)] = 1;
    }
  }
  parts.push(bigram);

  const center = new Float32Array(NUM_CHARS);
  const centerIdx = Math.floor(contextLen / 2);
  if (centerIdx >= 0 && centerIdx < contextLen) {
    const c = context[centerIdx]!;
    if (c in CHAR_TO_IDX) center[CHAR_TO_IDX[c]!] = 2;
  }
  parts.push(center);

  const phonePos = new Float32Array(8);
  if (phonemePos != null) {
    phonePos[0] = phonemePos;
    phonePos[1] = 1 - phonemePos;
    phonePos[2] = Math.sin(Math.PI * phonemePos);
    phonePos[3] = Math.cos(Math.PI * phonemePos);
    phonePos[4] = phonemePos < 0.25 ? 1 : 0;
    phonePos[5] = phonemePos > 0.75 ? 1 : 0;
    if (nPhonemes != null) {
      phonePos[6] = 1 / Math.max(nPhonemes, 1);
      phonePos[7] = nPhonemes <= 2 ? 1 : 0;
    }
  }
  parts.push(phonePos);

  const prev = new Float32Array(NUM_PHONEMES);
  if (previousPhone && previousPhone in PHONEME_TO_IDX) {
    prev[PHONEME_TO_IDX[previousPhone]!] = 1;
  }
  parts.push(prev);

  const focus = new Float32Array(4);
  if (nPhonemes != null && phonemePos != null) {
    focus[0] = nPhonemes <= 3 ? 1 : 0;
    if (nPhonemes === 2) {
      focus[1] = phonemePos < 0.5 ? 1 : -1;
    }
    const slotIdx = nPhonemes > 1 ? Math.trunc(phonemePos * (nPhonemes - 1)) : 0;
    focus[2] = (slotIdx % 3) / 2.0 - 0.5;
    let nLetters = 0;
    for (let i = 0; i < letterContext.length; i++) {
      if (letterContext[i] !== "_") nLetters++;
    }
    focus[3] = Math.min(nLetters / 10, 1);
  }
  parts.push(focus);

  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Float32Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  void contextSize;
  return out;
}

export function buildLetterContext(word: string, letterPos: number, contextSize: number): string {
  let s = "";
  for (let off = -contextSize; off <= contextSize; off++) {
    const i = letterPos + off;
    s += i >= 0 && i < word.length ? word[i]! : "_";
  }
  return s;
}

export function slotGeometry(word: string, pIdx: number, nPhonemes: number) {
  const nLetters = word.length;
  let letterPos: number;
  let phonemePos: number;
  if (nPhonemes === 1) {
    letterPos = Math.floor(nLetters / 2);
    phonemePos = 0.5;
  } else {
    letterPos = Math.round((pIdx * (nLetters - 1)) / (nPhonemes - 1));
    phonemePos = pIdx / (nPhonemes - 1);
  }
  letterPos = Math.max(0, Math.min(letterPos, nLetters - 1));
  return { letterPos, phonemePos };
}
