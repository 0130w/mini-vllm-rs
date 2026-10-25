use std::io::{Error, ErrorKind, Result};

use crate::{
    file_loader::SafetensorsLoader,
    model::Tensor,
    models::qwen3_5::config::{LayerType, TextConfig},
};

// see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L811
pub(crate) struct MlpWeights {
    pub(crate) gate_proj: Tensor,
    pub(crate) up_proj: Tensor,
    pub(crate) down_proj: Tensor,
}

// see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L736
pub(crate) struct FullAttentionWeights {
    pub(crate) q_proj: Tensor,
    pub(crate) k_proj: Tensor,
    pub(crate) v_proj: Tensor,
    pub(crate) o_proj: Tensor,
    pub(crate) q_norm: Tensor,
    pub(crate) k_norm: Tensor,
}

pub(crate) struct GatedDeltaNetWeights {
    pub(crate) conv1d: Tensor,
    pub(crate) dt_bias: Tensor,
    pub(crate) a_log: Tensor,
    pub(crate) norm: Tensor,
    pub(crate) out_proj: Tensor,
    pub(crate) in_proj_qkv: Tensor,
    pub(crate) in_proj_z: Tensor,
    pub(crate) in_proj_b: Tensor,
    pub(crate) in_proj_a: Tensor,
}

pub(crate) enum AttentionWeights {
    Full(FullAttentionWeights),
    Linear(GatedDeltaNetWeights),
}

pub(crate) struct DecoderLayerWeights {
    pub(crate) layer_norm: Tensor,
    pub(crate) attention: AttentionWeights,
    pub(crate) post_layer_norm: Tensor,
    pub(crate) mlp: MlpWeights,
}

pub(crate) struct Qwen3_5ModelWeights {
    pub(crate) embed_tokens: Tensor,
    pub(crate) layers: Vec<DecoderLayerWeights>,
    pub(crate) norm: Tensor,
}

// see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1209
pub struct ModelWeights {
    pub(crate) qwen3_5_model_weights: Qwen3_5ModelWeights,
}

pub(crate) fn load_weights(
    loader: &mut SafetensorsLoader,
    config: &TextConfig,
) -> Result<ModelWeights> {
    if config.layer_types.len() != config.num_hidden_layers {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "layer_types length does not match num_hidden_layers",
        ));
    }

    // Keep this restriction until loading and using a separate LM head is implemented.
    if !config.tie_word_embeddings {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "untied output embeddings are not supported yet",
        ));
    }

    // Keep this restriction until attention projection biases are loaded and applied.
    if config.attention_bias {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "attention bias is not supported yet",
        ));
    }

    let root = "model.language_model";
    let embed_tokens = loader.load_tensor(&format!("{root}.embed_tokens.weight"))?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for (idx, layer_type) in config.layer_types.iter().enumerate() {
        let prefix = format!("{root}.layers.{idx}");
        let mut load = |suffix: &str| loader.load_tensor(&format!("{prefix}.{suffix}"));
        let attention = match layer_type {
            LayerType::FullAttention => AttentionWeights::Full(FullAttentionWeights {
                q_proj: load("self_attn.q_proj.weight")?,
                k_proj: load("self_attn.k_proj.weight")?,
                v_proj: load("self_attn.v_proj.weight")?,
                o_proj: load("self_attn.o_proj.weight")?,
                q_norm: load("self_attn.q_norm.weight")?,
                k_norm: load("self_attn.k_norm.weight")?,
            }),

            LayerType::LinearAttention => AttentionWeights::Linear(GatedDeltaNetWeights {
                conv1d: load("linear_attn.conv1d.weight")?,
                dt_bias: load("linear_attn.dt_bias")?,
                a_log: load("linear_attn.A_log")?,
                norm: load("linear_attn.norm.weight")?,
                out_proj: load("linear_attn.out_proj.weight")?,
                in_proj_qkv: load("linear_attn.in_proj_qkv.weight")?,
                in_proj_z: load("linear_attn.in_proj_z.weight")?,
                in_proj_b: load("linear_attn.in_proj_b.weight")?,
                in_proj_a: load("linear_attn.in_proj_a.weight")?,
            }),
        };
        let mlp = MlpWeights {
            gate_proj: load("mlp.gate_proj.weight")?,
            up_proj: load("mlp.up_proj.weight")?,
            down_proj: load("mlp.down_proj.weight")?,
        };

        let layer = DecoderLayerWeights {
            layer_norm: load("input_layernorm.weight")?,
            attention,
            post_layer_norm: load("post_attention_layernorm.weight")?,
            mlp,
        };
        layers.push(layer);
    }

    let norm = loader.load_tensor(&format!("{root}.norm.weight"))?;
    Ok(ModelWeights {
        qwen3_5_model_weights: Qwen3_5ModelWeights {
            embed_tokens,
            layers,
            norm,
        },
    })
}
