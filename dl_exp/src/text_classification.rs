use candle::{DType, Result, Tensor,Device};
use candle_nn::{ Linear, Embedding, Module,  VarBuilder, VarMap};


fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}

pub trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

struct LinearModel {
    linear: Linear,
}

impl Model for LinearModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear = linear_z(100, 10, vs)?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

#[derive(Debug)]
struct Mlp {
    emb: Embedding,
    ln1: Linear,
    ln2: Linear,
}

impl Model for Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let emb = candle_nn::embedding(100, 50, vs.pp("emb"))?;
        let ln1 = candle_nn::linear(100, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, 10, vs.pp("ln2"))?;
        Ok(Self { emb,ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs  = self.emb.forward(xs)?;
        let xs = self.ln1.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}



pub fn inference_linear() -> Result<()> {
    let device = Device::new_metal(0)?;
    let a = Tensor::randn(0f32, 1., (1, 100), &device)?;
    println!("tenor shape : {:?}",a.shape());



    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Mlp::new(vs.clone())?;

    println!("Model : {:?} ", model);
    let out = model.forward(&a)?;
    println!("Model output {:?}", out);

    Ok(())
}

