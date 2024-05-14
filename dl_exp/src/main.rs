mod text_classification;
use anyhow::Ok;

fn main() -> anyhow::Result<()> {
    text_classification::inference_linear();

    Ok(())
}