use std::{
    collections::HashMap,
    fs::File,
    io::{Error, ErrorKind, Read, Result, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use cuda_core::{CudaContext, DeviceBuffer};

use crate::model::Tensor;

#[derive(serde::Deserialize, Debug)]
pub(crate) struct TensorInfo {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    #[serde(rename = "data_offsets")]
    pub(crate) offsets: [u64; 2], // [start, end), not includes end pos
}

pub(crate) struct SafetensorsLoader {
    file: File,
    tensors: HashMap<String, TensorInfo>, // mapping from tensor name to metadata
    data_start: u64,                      // start position of tensor data in file
    file_len: u64,                        // file size
    ctx: Arc<CudaContext>,
}

impl SafetensorsLoader {
    pub(crate) fn open(path: impl AsRef<Path>, ctx: Arc<CudaContext>) -> Result<Self> {
        let invalid = |msg: &str| Error::new(ErrorKind::InvalidData, msg);

        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();

        // 8 bytes starts from file: the number of bytes of header in le order
        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)?;
        let header_len = u64::from_le_bytes(length_bytes);

        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| invalid("header size overflow"))?;

        if data_start > file_len {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "header exceeds file length",
            ));
        }

        let header_len =
            usize::try_from(header_len).map_err(|_| invalid("header size exceeds usize"))?;

        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;

        let mut header: serde_json::Value = serde_json::from_slice(&header_bytes)?;

        if let Some(object) = header.as_object_mut() {
            object.remove("__metadata__");
        }

        let tensors: HashMap<String, TensorInfo> = serde_json::from_value(header)?;

        Ok(Self {
            file,
            tensors,
            data_start,
            file_len,
            ctx,
        })
    }

    fn read_tensor_bytes(&mut self, name: &str, offsets: [u64; 2]) -> Result<Vec<u8>> {
        let invalid = |msg: &str| Error::new(ErrorKind::InvalidData, format!("{name}: {msg}"));
        let [start, end] = offsets;

        let byte_len = end
            .checked_sub(start)
            .ok_or_else(|| invalid("end offset precedes start offset"))?;

        let file_start = self
            .data_start
            .checked_add(start)
            .ok_or_else(|| invalid("start offset overflow"))?;

        let file_end = self
            .data_start
            .checked_add(end)
            .ok_or_else(|| invalid("end offset overflow"))?;

        // file maybe chunked by mistake
        if file_end > self.file_len {
            return Err(invalid("tensor data exceeds file length"));
        }

        let byte_len =
            usize::try_from(byte_len).map_err(|_| invalid("tensor byte length exceeds usize"))?;

        self.file.seek(SeekFrom::Start(file_start))?;

        let mut bytes = vec![0u8; byte_len];

        self.file.read_exact(&mut bytes)?;

        Ok(bytes)
    }

    pub(crate) fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        let invalid = |msg: &str| Error::new(ErrorKind::InvalidData, format!("{name}: {msg}"));

        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid("tensor not found"))?;
        let dtype = info.dtype.clone();
        let shape: Vec<usize> = info
            .shape
            .iter()
            .map(|&dim| usize::try_from(dim).map_err(|_| invalid("dimension exceeds usize")))
            .collect::<Result<_>>()?;
        let element_size = match dtype.as_str() {
            "BF16" => 2,
            "F32" => 4,
            _ => return Err(invalid("unsupported dtype")),
        };
        let bytes = self.read_tensor_bytes(name, info.offsets)?;

        if bytes.len() % element_size != 0 {
            return Err(invalid("byte length is not divisible by element size"));
        }

        let stream = self.ctx.default_stream();
        match dtype.as_str() {
            "BF16" => {
                let values: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                drop(bytes);
                let buffer = DeviceBuffer::from_host(&stream, &values)
                    .map_err(|e| Error::other(format!("{name}: {e}")))?;
                stream
                    .synchronize()
                    .map_err(|e| Error::other(format!("{name}: {e}")))?;
                Ok(Tensor::from_bf16_bits(buffer, shape))
            }
            "F32" => {
                let values: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                drop(bytes);
                let buffer = DeviceBuffer::from_host(&stream, &values)
                    .map_err(|e| Error::other(format!("{name}: {e}")))?;
                stream
                    .synchronize()
                    .map_err(|e| Error::other(format!("{name}: {e}")))?;
                Ok(Tensor::from_f32_bits(buffer, shape))
            }
            _ => unreachable!(),
        }
    }
}
