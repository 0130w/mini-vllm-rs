#[derive(Clone)]
pub struct Tensor;

pub trait Model {
    /// One unpadded text request: token_ids [seq_len] -> logits [seq_len, vocab_size].
    fn forward(&mut self, token_ids: &[u32]) -> Tensor;
}
