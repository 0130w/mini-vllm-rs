use crate::{model::Tensor, weights::TensorInfo};

// see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L811
pub(crate) struct MlpWeights {
    pub(crate) gate_proj: TensorInfo,
    pub(crate) up_proj: TensorInfo,
    pub(crate) down_proj: TensorInfo,
}

// see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L736
pub(crate) struct FullAttentionWeights {
    pub(crate) q_proj: TensorInfo,
    pub(crate) k_proj: TensorInfo,
    pub(crate) v_proj: TensorInfo,
    pub(crate) o_proj: TensorInfo,
    pub(crate) q_norm: TensorInfo,
    pub(crate) k_norm: TensorInfo,
}

pub(crate) struct GatedDeltaNetWeights {
    conv1d: TensorInfo,
    pub(crate) dt_bias: TensorInfo,
    pub(crate) a_log: TensorInfo,
    pub(crate) norm: TensorInfo,
    pub(crate) out_proj: TensorInfo,
    pub(crate) in_proj_qkv: TensorInfo,
    pub(crate) in_proj_z: TensorInfo,
    pub(crate) in_proj_b: TensorInfo,
    pub(crate) in_proj_a: TensorInfo,
}

pub(crate) enum AttentionWeights {
    Full(FullAttentionWeights),
    Linear(GatedDeltaNetWeights),
}

pub(crate) struct DecoderLayer {
    pub(crate) layer_norm: TensorInfo,
    pub(crate) attention: AttentionWeights,
    pub(crate) post_layer_norm: TensorInfo,
    pub(crate) mlp: MlpWeights,
}

pub(crate) struct Qwen3_5ModelWeights {
    pub(crate) embed_tokens: Tensor,
}

// see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1209
pub struct ModelWeights {
    pub(crate) qwen3_5_model_weights: Qwen3_5ModelWeights,
}
