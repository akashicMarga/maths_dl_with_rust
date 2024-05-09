use anyhow::Ok;
use candle::{Device, Tensor};
use std::time::Instant;

pub fn cosine_similarity() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    // Adjusting both tensors to have the same dimensions
    let a = Tensor::randn(0f32, 1., (1, 6), &device)?;
    let b = Tensor::randn(0f32, 1., (1, 6), &device)?;

    println!("a:\n{a}\n");
    println!("b:\n{b}\n");

    let start = Instant::now();
    let sum_ab: f32 = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let sum_aa: f32 = (&a * &a)?.sum_all()?.to_scalar::<f32>()?;
    let sum_bb: f32 = (&b * &b)?.sum_all()?.to_scalar::<f32>()?;
    let cosine_similarity = sum_ab / (sum_aa * sum_bb).sqrt();
    let duration = start.elapsed();

    println!("Cosine similarity score c:\n{cosine_similarity}\nTime taken: {:?}", duration);

    Ok(())
}

