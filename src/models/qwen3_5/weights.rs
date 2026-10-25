use crate::model::Tensor;

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
