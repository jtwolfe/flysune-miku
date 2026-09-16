import { PHONEME_LIST, PHONEME_TO_IDX, type Phoneme } from "./types";

export type PhonemeInfo = {
  symbol: Phoneme;
  ipa: string;
  kind: "vowel" | "consonant" | "semivowel";
  example: string;
  description: string;
};

export const PHONEME_INVENTORY: Record<Phoneme, PhonemeInfo> = {
  AA: { symbol: "AA", ipa: "ɑ", kind: "vowel", example: "odd", description: "open back unrounded" },
  AE: { symbol: "AE", ipa: "æ", kind: "vowel", example: "at", description: "near-open front unrounded" },
  AH: { symbol: "AH", ipa: "ʌ", kind: "vowel", example: "hut", description: "open-mid back unrounded" },
  AO: { symbol: "AO", ipa: "ɔ", kind: "vowel", example: "ought", description: "open-mid back rounded" },
  EH: { symbol: "EH", ipa: "ɛ", kind: "vowel", example: "ed", description: "open-mid front unrounded" },
  ER: { symbol: "ER", ipa: "ɝ", kind: "vowel", example: "hurt", description: "r-colored mid central" },
  IH: { symbol: "IH", ipa: "ɪ", kind: "vowel", example: "it", description: "near-close front unrounded" },
  IY: { symbol: "IY", ipa: "i", kind: "vowel", example: "eat", description: "close front unrounded" },
  UH: { symbol: "UH", ipa: "ʊ", kind: "vowel", example: "hood", description: "near-close back rounded" },
  UW: { symbol: "UW", ipa: "u", kind: "vowel", example: "two", description: "close back rounded" },
  AW: { symbol: "AW", ipa: "aʊ", kind: "vowel", example: "cow", description: "diphthong" },
  AY: { symbol: "AY", ipa: "aɪ", kind: "vowel", example: "hide", description: "diphthong" },
  EY: { symbol: "EY", ipa: "eɪ", kind: "vowel", example: "ate", description: "diphthong" },
  OW: { symbol: "OW", ipa: "oʊ", kind: "vowel", example: "oat", description: "diphthong" },
  OY: { symbol: "OY", ipa: "ɔɪ", kind: "vowel", example: "toy", description: "diphthong" },
  P: { symbol: "P", ipa: "p", kind: "consonant", example: "pea", description: "voiceless bilabial stop" },
  B: { symbol: "B", ipa: "b", kind: "consonant", example: "bee", description: "voiced bilabial stop" },
  T: { symbol: "T", ipa: "t", kind: "consonant", example: "tea", description: "voiceless alveolar stop" },
  D: { symbol: "D", ipa: "d", kind: "consonant", example: "dee", description: "voiced alveolar stop" },
  K: { symbol: "K", ipa: "k", kind: "consonant", example: "key", description: "voiceless velar stop" },
  G: { symbol: "G", ipa: "ɡ", kind: "consonant", example: "gee", description: "voiced velar stop" },
  CH: { symbol: "CH", ipa: "tʃ", kind: "consonant", example: "cheese", description: "voiceless postalveolar affricate" },
  JH: { symbol: "JH", ipa: "dʒ", kind: "consonant", example: "gee", description: "voiced postalveolar affricate" },
  F: { symbol: "F", ipa: "f", kind: "consonant", example: "fee", description: "voiceless labiodental fricative" },
  V: { symbol: "V", ipa: "v", kind: "consonant", example: "vee", description: "voiced labiodental fricative" },
  TH: { symbol: "TH", ipa: "θ", kind: "consonant", example: "thin", description: "voiceless dental fricative" },
  DH: { symbol: "DH", ipa: "ð", kind: "consonant", example: "then", description: "voiced dental fricative" },
  S: { symbol: "S", ipa: "s", kind: "consonant", example: "sea", description: "voiceless alveolar fricative" },
  Z: { symbol: "Z", ipa: "z", kind: "consonant", example: "zee", description: "voiced alveolar fricative" },
  SH: { symbol: "SH", ipa: "ʃ", kind: "consonant", example: "she", description: "voiceless postalveolar fricative" },
  ZH: { symbol: "ZH", ipa: "ʒ", kind: "consonant", example: "measure", description: "voiced postalveolar fricative" },
  HH: { symbol: "HH", ipa: "h", kind: "consonant", example: "he", description: "voiceless glottal fricative" },
  M: { symbol: "M", ipa: "m", kind: "consonant", example: "me", description: "bilabial nasal" },
  N: { symbol: "N", ipa: "n", kind: "consonant", example: "knee", description: "alveolar nasal" },
  NG: { symbol: "NG", ipa: "ŋ", kind: "consonant", example: "sing", description: "velar nasal" },
  L: { symbol: "L", ipa: "l", kind: "consonant", example: "lee", description: "alveolar lateral" },
  R: { symbol: "R", ipa: "ɹ", kind: "consonant", example: "ray", description: "alveolar approximant" },
  W: { symbol: "W", ipa: "w", kind: "semivowel", example: "way", description: "labio-velar approximant" },
  Y: { symbol: "Y", ipa: "j", kind: "semivowel", example: "yes", description: "palatal approximant" },
};

export const VOWELS = new Set(
  PHONEME_LIST.filter((p) => PHONEME_INVENTORY[p].kind === "vowel"),
);
export const CONSONANTS = new Set(
  PHONEME_LIST.filter((p) => PHONEME_INVENTORY[p].kind === "consonant"),
);
export const SEMIVOWELS = new Set(
  PHONEME_LIST.filter((p) => PHONEME_INVENTORY[p].kind === "semivowel"),
);
export const STOPS = new Set(["P", "B", "T", "D", "K", "G"]);
export const AFFRICATES = new Set(["CH", "JH"]);
export const FRICATIVES = new Set(["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"]);
export const NASALS = new Set(["M", "N", "NG"]);
export const LIQUIDS = new Set(["L", "R"]);
export const VOICED_CONSONANTS = new Set([
  "B", "D", "G", "JH", "V", "DH", "Z", "ZH", "M", "N", "NG", "L", "R",
]);
export const DIPHTHONGS = new Set(["AW", "AY", "EY", "OW", "OY"]);

export function stripStress(phoneme: string): string {
  return phoneme.replace(/[012]$/, "");
}

export function asPhoneme(raw: string): Phoneme {
  const p = stripStress(raw.toUpperCase());
  if (!(p in PHONEME_TO_IDX)) {
    throw new Error(`Unknown phoneme: ${raw}`);
  }
  return p as Phoneme;
}

export function isPhoneme(raw: string): raw is Phoneme {
  return raw in PHONEME_TO_IDX;
}
