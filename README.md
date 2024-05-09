# Rust Math and Deep Learning Fundamentals

## Overview
This project is a personal exploration into the fundamentals of mathematics and deep learning, implemented in Rust. The aim is to revisit common algorithms and concepts while simultaneously deepening understanding of Rust's capabilities, particularly in terms of hardware-specific accelerations.

## Objectives
- **Refresh Fundamental Concepts**: Revisit essential mathematics and deep learning principles.
- **Experiment with Rust**: Implement algorithms in Rust to leverage its performance and safety features.
- **Explore Hardware Acceleration**: Understand and utilize hardware-specific optimizations available in Rust.

## Project Structure
The project is organized into various modules, each focusing on different algorithms and concepts. Here is a tentative structure:

- **Math Basics**: Basic mathematical operations and theories.
- **Linear Algebra**: Vectors, matrices, and operations that are fundamental to deep learning.
- **Calculus**: Differentiation and integration relevant to optimization problems in deep learning.
- **Algorithms**: Implementation of common algorithms used in data science and machine learning.
- **Hardware Acceleration**: Exploration of Rust's ability to leverage CPU and GPU for computational speedups.

## Getting Started
To get started with this project, you will need to have Rust installed on your machine. Follow these steps:

1. Install Rust:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rust-math-deep-learning.git
   ```
3. Navigate to the project directory:
   ```bash
   cd rust-math-deep-learning
   ```
   navigate to either dl_exp or maths_exp and run the following command:
4. Build the project:
   ```bash
   cargo build
   ```
5. Run experiments:
   ```bash
   cargo run --release --features metal -- algorithm_name
   ```



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

