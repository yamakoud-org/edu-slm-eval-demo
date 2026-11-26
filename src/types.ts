export type ModelKey = 'baseline' | 'prompt_only' | 'weight_edited';

export type PrincipleKey =
  | 'active_learning'
  | 'cognitive_load'
  | 'adapt_learner'
  | 'curiosity'
  | 'metacognition';

export interface PrincipleMeta {
  key: PrincipleKey;
  label: string;
  description: string;
}

export interface ModelSummary {
  label: string;
  overall_score: number;
  principles: Record<PrincipleKey, number>;
  jailbreak_resistance_rate: number;
  jailbreak_resisted_count: number;
  jailbreak_total: number;
  regex_non_leak_rate: number;
  question_rate: number;
  avg_latency_sec: number;
  avg_gen_tokens: number;
  avg_repetition_ratio: number;
  by_category: Record<string, number>;
}

export interface EvalModelOutput {
  response: string;
  gen_tokens: number;
  latency_sec: number;
  has_question: boolean;
  matched_forbidden_patterns: string[];
  repetition_ratio: number;
  scores: Record<PrincipleKey, number>;
  overall_score: number;
  jailbreak_resisted: boolean | null;
}

export interface EvalItem {
  id: string;
  category: string;
  is_jailbreak: boolean;
  title: string;
  student_prompt: string;
  expected_pedagogy_focus: string;
  outputs: Record<ModelKey, EvalModelOutput>;
  weight_edited_neutral_preview: {
    response: string;
    has_question: boolean;
    matched_forbidden_patterns: string[];
  };
}

export interface EvalResultsData {
  meta: {
    base_model: string;
    edited_model: string;
    gpu: string;
    decoding: string;
    rubric_reference: string;
    num_questions: number;
    num_jailbreak_questions: number;
  };
  principles: PrincipleMeta[];
  summary: Record<ModelKey, ModelSummary>;
  diagnostic_weight_only_neutral: {
    label: string;
    question_rate: number;
    regex_non_leak_rate: number;
    avg_repetition_ratio: number;
    note: string;
  };
  items: EvalItem[];
}

export interface ProbingLayerStat {
  layer: number;
  delta_norm: number;
  resid_norm: number;
  rel_delta_norm: number;
  cos_syc_vs_neg_delta: number;
  cohen_d: number;
  cv_acc: number;
  mlp_key_stability_mean: number;
  mlp_key_stability_std: number;
  attn_key_stability_mean: number;
  attn_key_stability_std: number;
  cohen_d_math_jailbreak: number;
  cohen_d_adversarial_override: number;
}

export interface SurgeryLayerMetric {
  layer: number;
  beta_o: number;
  beta_d: number;
  alpha_o: number;
  alpha_d: number;
  o_proj_rel_fro: number;
  down_proj_rel_fro: number;
  clamped: boolean;
  o_proj_syc_energy_before: number;
  o_proj_syc_energy_after: number;
  down_proj_syc_energy_before: number;
  down_proj_syc_energy_after: number;
}

export interface DialogueTurn {
  turn: number;
  user: string;
  assistant: string;
  metrics: {
    char_len: number;
    has_question_mark: boolean;
    repetition_ratio: number;
    non_jp_ascii_junk_ratio: number;
    leak_signals: string[];
  };
}

export interface PersonaConversation {
  persona_id: string;
  label: string;
  system_mode: 'edu' | 'neutral';
  system_prompt: string;
  turns: DialogueTurn[];
}

export interface SurgeryCycle {
  id: string;
  version: string;
  title: string;
  status: 'baseline' | 'over_intervened' | 'under_intervened' | 'final_optimal';
  layers: number[];
  key_mode: string;
  diagnosis: string;
  prescription: string;
  metrics: { layers: SurgeryLayerMetric[] } | null;
  recipe: Record<string, unknown> | null;
  qualitative: {
    total_leaks: number;
    edu_leaks: number;
    neutral_leaks: number;
    question_rate_all: number;
    question_rate_edu: number;
    avg_repetition: number;
    max_repetition: number;
    conversations: PersonaConversation[];
  };
}

export interface SurgeryReportData {
  meta: {
    model_id: string;
    architecture: {
      num_hidden_layers: number;
      hidden_size: number;
      intermediate_size: number;
      o_proj_bias: boolean;
      down_proj_bias: boolean;
      dtype: string;
    };
    method: {
      abliteration_formula: string;
      weight_baking_formula: string;
      zero_gradient_training: boolean;
      modified_layers_final: number[];
      max_rel_frobenius_o_proj: number;
      max_rel_frobenius_down_proj: number;
    };
  };
  probing: {
    v1_summary: {
      pairs: number;
      window_tokens: number;
      best_cohen_d: number;
      best_cv_acc: number;
      best_rel_delta: number;
    };
    v2_summary: {
      pairs: number;
      window_tokens: number;
      best_cohen_d: number;
      best_cv_acc: number;
      best_rel_delta: number;
    };
    layers_v2: ProbingLayerStat[];
  };
  cycles: SurgeryCycle[];
}
