use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
}

/// omit undeclaed fields by default
/// see models/qwen-3.5-0.8B/config.json
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<LayerType>,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub attn_output_gate: bool,

    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,

    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_parameters: RopeConfig,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5Config {
    pub text_config: TextConfig,
}
