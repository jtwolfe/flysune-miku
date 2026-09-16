import { PHONEME_LIST, type Phoneme, type SlotCountSource } from "./types";
import { isPhoneme } from "./phonemes";

type RecordPtr = { word: string; phones: Phoneme[] };

let records: RecordPtr[] | null = null;

function decodeCmudict(buf: ArrayBuffer): RecordPtr[] {
  const view = new DataView(buf);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (magic !== "FLYD") throw new Error("Bad cmudict magic");
  const n = view.getUint32(4, true);
  const bytes = new Uint8Array(buf);
  const out: RecordPtr[] = [];
  let o = 8;
  const dec = new TextDecoder("ascii");
  for (let i = 0; i < n; i++) {
    const wl = bytes[o++];
    const pl = bytes[o++];
    const word = dec.decode(bytes.subarray(o, o + wl));
    o += wl;
    const phones: Phoneme[] = [];
    for (let j = 0; j < pl; j++) {
      phones.push(PHONEME_LIST[bytes[o++]]!);
    }
    out.push({ word, phones });
  }
  return out;
}

function bisect(word: string): Phoneme[] | null {
  if (!records) return null;
  let lo = 0;
  let hi = records.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const w = records[mid]!.word;
    if (w === word) return records[mid]!.phones;
    if (w < word) lo = mid + 1;
    else hi = mid - 1;
  }
  return null;
}

export async function loadLexicon(): Promise<void> {
  if (records) return;
  const res = await fetch("/models/cmudict.bin");
  if (!res.ok) throw new Error("Failed to load CMUdict");
  records = decodeCmudict(await res.arrayBuffer());
}

export function isKnownWord(word: string): boolean {
  return bisect(word.toLowerCase()) !== null;
}

const DIGRAPHS: [string, Phoneme][] = [
  ["th", "TH"],
  ["sh", "SH"],
  ["ch", "CH"],
  ["ng", "NG"],
  ["ee", "IY"],
  ["oo", "UW"],
  ["ou", "AW"],
  ["ai", "EY"],
  ["ay", "EY"],
];

const LETTER: Record<string, Phoneme> = {
  a: "AE", e: "EH", i: "IH", o: "AA", u: "AH",
  b: "B", c: "K", d: "D", f: "F", g: "G",
  h: "HH", j: "JH", k: "K", l: "L", m: "M",
  n: "N", p: "P", q: "K", r: "R", s: "S",
  t: "T", v: "V", w: "W", x: "K", y: "IY",
  z: "Z",
};

export function simpleG2p(word: string): Phoneme[] {
  const w = word.toLowerCase();
  const phones: Phoneme[] = [];
  let i = 0;
  while (i < w.length) {
    let hit = false;
    if (i + 1 < w.length) {
      const pair = w.slice(i, i + 2);
      for (const [d, p] of DIGRAPHS) {
        if (pair === d) {
          phones.push(p);
          i += 2;
          hit = true;
          break;
        }
      }
    }
    if (hit) continue;
    const p = LETTER[w[i]!];
    if (p) phones.push(p);
    i += 1;
  }
  return phones.length ? phones : ["AH"];
}

export function getPhonemes(word: string): { phones: Phoneme[]; source: SlotCountSource; known: boolean } {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  const fromDict = bisect(w);
  if (fromDict) return { phones: fromDict, source: "cmudict", known: true };
  const phones = simpleG2p(w).filter(isPhoneme);
  return { phones: phones.length ? phones : ["AH"], source: "simple", known: false };
}
