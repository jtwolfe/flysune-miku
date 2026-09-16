export const PHONEME_LIST = [
  "AA", "AE", "AH", "AO", "EH", "ER", "IH", "IY", "UH", "UW",
  "AW", "AY", "EY", "OW", "OY",
  "P", "B", "T", "D", "K", "G",
  "CH", "JH",
  "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH",
  "M", "N", "NG",
  "L", "R",
  "W", "Y",
] as const;

export type Phoneme = (typeof PHONEME_LIST)[number];

export const NUM_PHONEMES = PHONEME_LIST.length;

export const PHONEME_TO_IDX: Record<string, number> = Object.fromEntries(
  PHONEME_LIST.map((p, i) => [p, i]),
);

export type SlotCountSource = "cmudict" | "simple";

export type Vote = {
  phoneme: Phoneme;
  isYes: boolean;
  confidence: number;
  score: number;
  mbonYes: number;
  mbonNo: number;
};

export type SpeakerParams = {
  f0: number;
  f1Shift: number;
  f2Shift: number;
  f3Shift: number;
  noiseLevel: number;
  durationMs: number;
  attackMs: number;
  releaseMs: number;
  harmonicRolloff: number;
  prevPhoneF0Delta?: number[];
  prevPhoneDurDelta?: number[];
};

export type SlotTrace = {
  index: number;
  letterContext: string;
  letterPos: number;
  phonemePos: number;
  nPhonemes: number;
  previousPhone: Phoneme | null;
  referencePhone: Phoneme;
  pickerPhone: Phoneme;
  confidence: number;
  nKcActive: number;
  nKc: number;
  kcActive: number[];
  votes: Vote[];
  speaker: SpeakerParams & { phoneme: Phoneme; durationMs: number };
  crumb: Float32Array;
  startSample: number;
};

export type TokenTrace = {
  word: string;
  pauseS: number;
  isKnown: boolean;
  slotCountSource: SlotCountSource;
  referencePhonemes: Phoneme[];
  pickerPhonemes: Phoneme[];
  nMatch: number;
  slots: SlotTrace[];
};

export type SpeakTrace = {
  text: string;
  tokens: TokenTrace[];
  sampleRate: number;
  audio: Float32Array;
  nMatch: number;
  nTotal: number;
  architecture: string;
};

export type PickerManifest = {
  version: number;
  phonemes: Phoneme[];
  inputDim: number;
  nPn: number;
  nKc: number;
  kcSparsity: number;
  contextSize: number;
  voteTemperature: number;
  cue: {
    use_position_features: boolean;
    use_previous_phone: boolean;
    use_focus_features: boolean;
    position_dim: number;
    previous_phone_dim: number;
    focus_dim: number;
  };
  wiring: {
    mode: string;
    n_pn: number;
    n_kc: number;
    source: string;
    hash: string;
    sparsity: number;
    avg_pn_per_kc: number;
  };
  arrays: Record<
    string,
    { offset: number; shape: number[]; dtype: string; nbytes: number }
  >;
};
