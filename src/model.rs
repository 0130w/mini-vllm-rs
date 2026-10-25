use std::sync::Arc;

use cuda_core::DeviceBuffer;

#[derive(Clone)]
enum Storage {
    BF16(Arc<DeviceBuffer<u16>>),
    FP32(Arc<DeviceBuffer<f32>>),
}

impl Storage {
    fn len(&self) -> usize {
        match self {
            Self::BF16(buffer) => buffer.len(),
            Self::FP32(buffer) => buffer.len(),
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
}

impl Tensor {
    fn validate_shape(&self) {
        let numel = if self.shape.contains(&0) {
            0
        } else {
            self.shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .expect("tensor shape element count overflow")
        };
        assert_eq!(
            numel,
            self.storage.len(),
            "tensor shape {:?} does not match buffer length",
            self.shape
        )
    }

    pub fn from_bf16_bits(buffer: DeviceBuffer<u16>, shape: Vec<usize>) -> Self {
        let tensor = Self {
            storage: Storage::BF16(Arc::new(buffer)),
            shape,
        };
        tensor.validate_shape();
        tensor
    }

    pub fn from_f32_bits(buffer: DeviceBuffer<f32>, shape: Vec<usize>) -> Self {
        let tensor = Self {
            storage: Storage::FP32(Arc::new(buffer)),
            shape,
        };
        tensor.validate_shape();
        tensor
    }

    pub fn as_bf16_bits(&self) -> &DeviceBuffer<u16> {
        match &self.storage {
            Storage::BF16(buffer) => buffer.as_ref(),
            Storage::FP32(_) => panic!("expected BF16 tensor"),
        }
    }

    pub fn as_fp32_bits(&self) -> &DeviceBuffer<f32> {
        match &self.storage {
            Storage::FP32(buffer) => buffer.as_ref(),
            Storage::BF16(_) => panic!("expected FP32 tensor"),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub trait Model {
    /// One unpadded text request: token_ids [seq_len] -> logits [seq_len, vocab_size].
    fn forward(&mut self, token_ids: &[u32]) -> Tensor;
}
