use crate::model::{Model, Tensor};

struct Embedding {}

impl Embedding {
    /// [seq_len] token IDs -> [seq_len, hidden_size].
    fn forward(&self, _token_ids: &[u32]) -> Tensor {
        unimplemented!("embedding lookup")
    }

    /// Project hidden states to logits using the same embedding weight.
    fn project_to_vocab(&self, _hidden: &Tensor) -> Tensor {
        unimplemented!("output projection with shared embedding weight")
    }
}

struct Qwen3_5GatedDeltaNet {}

impl Qwen3_5GatedDeltaNet {
    fn forward(&mut self, _input: &Tensor) -> Tensor {
        unimplemented!("full-sequence gated delta net without persistent cache")
    }
}

struct Qwen3_5Attention {}

impl Qwen3_5Attention {
    fn forward(&mut self, _input: &Tensor, _positions: &RotaryPositions) -> Tensor {
        unimplemented!("causal full attention with RoPE and output gating")
    }
}

enum AttentionLayer {
    Linear(Qwen3_5GatedDeltaNet),
    Full(Qwen3_5Attention),
}

impl AttentionLayer {
    fn forward(&mut self, input: &Tensor, positions: &RotaryPositions) -> Tensor {
        match self {
            Self::Linear(attn) => attn.forward(input),
            Self::Full(attn) => attn.forward(input, positions),
        }
    }
}

struct Linear {}

impl Linear {
    fn forward(&self, _input: &Tensor) -> Tensor {
        unimplemented!("linear projection")
    }
}

struct Qwen3_5MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3_5MLP {
    fn forward(&self, input: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(input);
        let up = self.up_proj.forward(input);
        let activated = silu(&gate);
        let hidden = multiply(&activated, &up);
        self.down_proj.forward(&hidden)
    }
}

struct Qwen3_5DecoderLayer {
    attn: AttentionLayer,
    mlp: Qwen3_5MLP,
    input_norm: RMSNorm,
    post_attn_norm: RMSNorm,
}

impl Qwen3_5DecoderLayer {
    fn forward(&mut self, input: &Tensor, positions: &RotaryPositions) -> Tensor {
        // Borrow the residual: normalization must not overwrite input.
        let normalized = self.input_norm.forward(input);
        let mixed = self.attn.forward(&normalized, positions);
        let hidden = add(input, &mixed);

        let normalized = self.post_attn_norm.forward(&hidden);
        let mlp_output = self.mlp.forward(&normalized);
        add(&hidden, &mlp_output)
    }
}

struct RMSNorm {}

impl RMSNorm {
    fn forward(&self, _input: &Tensor) -> Tensor {
        unimplemented!("Qwen3.5 RMSNorm with 1 + weight scaling")
    }
}

struct RotaryPositions {
    cos: Tensor,
    sin: Tensor,
}

struct RotaryEmbedding {}

impl RotaryEmbedding {
    fn forward(&self, _position_ids: &[usize]) -> RotaryPositions {
        unimplemented!("RoPE cos/sin using the model configuration")
    }
}

struct Qwen3_5TextModel {
    embed_tokens: Embedding,
    decode_layers: Vec<Qwen3_5DecoderLayer>,
    norm: RMSNorm,
    rotary: RotaryEmbedding,
}

impl Qwen3_5TextModel {
    fn embed_tokens(&self, token_ids: &[u32]) -> Tensor {
        self.embed_tokens.forward(token_ids)
    }

    /// Embeddings -> final hidden states; positions supplied by the outer model.
    fn forward(&mut self, mut hidden: Tensor, position_ids: &[usize]) -> Tensor {
        let positions = self.rotary.forward(position_ids);
        for layer in &mut self.decode_layers {
            hidden = layer.forward(&hidden, &positions);
        }
        self.norm.forward(&hidden)
    }
}

pub struct Qwen3_5 {
    language_model: Qwen3_5TextModel,
}

impl Model for Qwen3_5 {
    fn forward(&mut self, token_ids: &[u32]) -> Tensor {
        // One unpadded text request; recompute the whole prefix without cache.
        let position_ids: Vec<usize> = (0..token_ids.len()).collect();
        let embeddings = self.language_model.embed_tokens(token_ids);
        let hidden = self.language_model.forward(embeddings, &position_ids);
        self.lm_head(&hidden)
    }
}

impl Qwen3_5 {
    /// [seq_len, hidden_size] -> [seq_len, vocab_size].
    /// The checkpoint ties the output weight to the embedding table.
    fn lm_head(&self, hidden: &Tensor) -> Tensor {
        self.language_model.embed_tokens.project_to_vocab(hidden)
    }
}

fn add(_a: &Tensor, _b: &Tensor) -> Tensor {
    unimplemented!("elementwise residual addition")
}

fn multiply(_a: &Tensor, _b: &Tensor) -> Tensor {
    unimplemented!("elementwise multiplication")
}

fn silu(_input: &Tensor) -> Tensor {
    unimplemented!("SiLU activation")
}
