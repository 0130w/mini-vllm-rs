use std::{
    collections::HashMap,
    fs::File,
    io::{Error, ErrorKind, Read, Result, Seek, SeekFrom},
    sync::Arc,
};

use cuda_core::{CudaContext, DeviceBuffer};

pub use crate::models::qwen3_5::weights::ModelWeights;
use crate::{model::Tensor, models::qwen3_5::weights::Qwen3_5ModelWeights};

#[derive(serde::Deserialize, Debug)]
pub(crate) struct TensorInfo {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    #[serde(rename = "data_offsets")]
    pub(crate) offsets: [u64; 2],
}

/// read safetensor files
pub fn load_weights(ctx: &Arc<CudaContext>) -> Result<ModelWeights> {
    let model_path: &str = "models/qwen-3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
    let mut safetensors_file = File::open(model_path)?;
    // safetensors file header_size is exactly 8 bytes
    let mut bytes = [0u8; 8];
    safetensors_file.read_exact(&mut bytes)?;
    let header_size = u64::from_le_bytes(bytes);
    let header_size = usize::try_from(header_size)
        .map_err(|_| std::io::Error::other("header size exceed usize"))?;
    // read header
    let mut bytes = vec![0u8; header_size];
    safetensors_file.read_exact(&mut bytes)?;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes)?;
    if let Some(object) = header.as_object_mut() {
        // "__metadata__" stores additional info
        object.remove("__metadata__");
    }
    let tensors: HashMap<String, TensorInfo> = serde_json::from_value(header)?;
    // offset starts from 0
    // an interesting question is what checks should be processed in `load_weights` while others should be processed in Tensor construction.
    let invalid = |msg: &str| Error::new(ErrorKind::InvalidData, msg);

    let embedding_info = tensors
        .get("model.language_model.embed_tokens.weight")
        .ok_or_else(|| invalid("missing embedding weight"))?;

    if embedding_info.dtype != "BF16" {
        return Err(invalid("embedding weight must be BF16"));
    }

    let shape: Vec<usize> = embedding_info
        .shape
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(|_| invalid("tensor dimension exceeds usize")))
        .collect::<Result<_>>()?;

    let [start, end] = embedding_info.offsets;
    let byte_len = end
        .checked_sub(start)
        .ok_or_else(|| invalid("invalid embedding offset"))?;

    let byte_len =
        usize::try_from(byte_len).map_err(|_| invalid("embedding byte length exceeds usize"))?;

    let data_start = 8u64
        .checked_add(header_size as u64)
        .ok_or_else(|| invalid("data start overflow"))?;

    let file_offset = data_start
        .checked_add(start)
        .ok_or_else(|| invalid("embedding file offset overflow"))?;

    let file_end = data_start
        .checked_add(end)
        .ok_or_else(|| invalid("embedding file end overflow"))?;

    if file_end > safetensors_file.metadata()?.len() {
        return Err(invalid("embedding data exceeds file length"));
    }

    safetensors_file.seek(SeekFrom::Start((file_offset)))?;

    let mut bytes = vec![0u8; byte_len];

    safetensors_file.read_exact(&mut bytes);

    // BF16 occupies 2 bytes
    let bits: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();

    drop(bytes);

    let stream = ctx.default_stream();
    let buffer =
        DeviceBuffer::from_host(&stream, &bits).map_err(|err| Error::other(err.to_string()))?;

    stream
        .synchronize()
        .map_err(|err| Error::other(err.to_string()))?;

    let embed_tokens = Tensor::from_bf16_bits(buffer, shape);

    Ok(ModelWeights {
        qwen3_5_model_weights: Qwen3_5ModelWeights { embed_tokens },
    })
}
