use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Result},
};

pub use crate::models::qwen3_5::weights::ModelWeights;

#[derive(serde::Deserialize, Debug)]
pub(crate) struct TensorInfo {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    #[serde(rename = "data_offsets")]
    pub(crate) offsets: [u64; 2],
}

/// read safetensor files
pub fn load_weights() -> Result<ModelWeights> {
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
    let data_size = tensors
        .values()
        .map(|tensor| tensor.offsets[1])
        .max()
        .unwrap_or(0);
    let data_size = usize::try_from(data_size)
        .map_err(|_| std::io::Error::other("header size exceed usize"))?;

    let mut data = vec![0u8; data_size];
    safetensors_file.read_exact(&mut data)?;

    #[cfg(feature = "cuda")]
    {}

    unimplemented!()
}
