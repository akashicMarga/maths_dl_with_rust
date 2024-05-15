mod matrix_mul; // This line imports the hello module
mod cosine_similarity;
mod pca;
mod kmeans;
mod utils;

use std::env;

fn main() -> anyhow::Result<()> {
    let program = env::args().nth(1).expect("No program specified");
    match program.as_str() {
        "matrix_mul" => matrix_mul::matrix_mul(),
        "cosine_similarity" => cosine_similarity::cosine_similarity(),
        "pca" => pca::run(),
        "kmeans" => kmeans::run(),
        _ => {
            println!("Unknown program: {}", program);
            Ok(())
        },
    }
}

