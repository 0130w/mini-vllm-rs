use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod embeddings {
    use cuda_device::{DisjointSlice, kernel, thread};

    #[kernel]
    pub fn qwen_3_5_embedding(
        token_ids: &[u32],
        embedding_weight: &[u16],
        mut output: DisjointSlice<u16>,
        hidden_size: usize,
    ) {
        // output[token_pos, col] = weight[token_ids[token_pos], col]
        // suppose one thread only move one element
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(elem) = output.get_mut(idx) {
            let token_pos = i / hidden_size;
            let col = i % hidden_size;
            let token_id = token_ids[token_pos] as usize;
            let weight_idx = token_id * hidden_size + col;
            *elem = embedding_weight[weight_idx];
        }
    }
}

pub fn embedding_forward(
    ctx: &Arc<CudaContext>,
    token_ids: &[u32],
    weight: &DeviceBuffer<u16>,
    hidden_size: usize,
) -> DeviceBuffer<u16> {
    assert!(hidden_size > 0);
    assert!(!token_ids.is_empty());
    assert_eq!(weight.len() % hidden_size, 0);

    let vocab_size = weight.len() / hidden_size;
    assert!(
        token_ids
            .iter()
            .all(|&token_id| (token_id as usize) < vocab_size)
    );

    let output_len = token_ids
        .len()
        .checked_mul(hidden_size)
        .expect("embedding output size overflow");

    let num_elems =
        u32::try_from(output_len).expect("embedding output exceeds launch helper range");

    let stream = ctx.default_stream();
    let token_ids_dev =
        DeviceBuffer::from_host(&stream, token_ids).expect("upload token IDs failed");
    let mut output =
        DeviceBuffer::zeroed(&stream, output_len).expect("allocate embedding output failed");

    let module = embeddings::load(ctx).expect("load embedding CUDA module failed");

    unsafe {
        module.qwen_3_5_embedding(
            &stream,
            LaunchConfig::for_num_elems(num_elems),
            &token_ids_dev,
            weight,
            &mut output,
            hidden_size,
        )
    }
    .expect("launch embedding kernel failed");

    stream
        .synchronize()
        .expect("embedding kernel execution failed");

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn embedding_lookup_small() {
        let ctx = CudaContext::new(0).expect("create CUDA context failed");
        let stream = ctx.default_stream();

        let weight_values = [
            1.0f32, 2.0, 3.0, 4.0, // token 0
            11.0, 12.0, 13.0, 14.0, // token 1
            21.0, 22.0, 23.0, 24.0, // token 2
        ];

        let weight_bits: Vec<u16> = weight_values
            .iter()
            .map(|&x| bf16::from_f32(x).to_bits())
            .collect();

        let weight_dev =
            DeviceBuffer::from_host(&stream, &weight_bits).expect("upload test weights failed");

        let token_ids = [2u32, 0, 2, 1];

        let output_dev = embedding_forward(&ctx, &token_ids, &weight_dev, 4);

        let output_host = output_dev
            .to_host_vec(&stream)
            .expect("download embedding output failed");

        let expected: Vec<u16> = token_ids
            .iter()
            .flat_map(|&id| {
                let start = id as usize * 4;
                weight_bits[start..start + 4].iter().copied()
            })
            .collect();

        assert_eq!(output_host, expected);
    }
}
