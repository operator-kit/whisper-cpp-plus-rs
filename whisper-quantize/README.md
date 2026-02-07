# whisper-quantize

Model quantization utilities for [whisper-cpp-plus](https://github.com/Code-Amp/whisper-cpp-plus). Reduces Whisper model sizes by 50-75% through quantization for deployment on resource-constrained devices.

## Usage

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};

// Quantize a model to 5-bit precision
ModelQuantizer::quantize_model_file(
    "models/ggml-base.bin",
    "models/ggml-base-q5_0.bin",
    QuantizationType::Q5_0,
)?;

// With progress tracking
ModelQuantizer::quantize_model_file_with_progress(
    "input.bin", "output.bin", QuantizationType::Q5_0,
    |progress| println!("{:.0}%", progress * 100.0),
)?;

// Check existing model quantization
if let Some(qtype) = ModelQuantizer::get_model_quantization_type("model.bin")? {
    println!("Quantized as: {}", qtype);
}

// Estimate output size
let estimated = ModelQuantizer::estimate_quantized_size("model.bin", QuantizationType::Q5_0)?;
```

## Quantization Types

### Standard (Q-series)

| Type | Size Reduction | Quality | Use Case |
|------|---------------|---------|----------|
| `Q4_0` | ~69% | Good | Mobile/embedded |
| `Q4_1` | ~65% | Good+ | Balanced |
| `Q5_0` | ~61% | Very Good | **Recommended** |
| `Q5_1` | ~57% | Very Good+ | Higher quality |
| `Q8_0` | ~31% | Excellent | Quality-critical |

### K-Quantization (K-series)

| Type | Size Reduction | Quality | Use Case |
|------|---------------|---------|----------|
| `Q2_K` | ~81% | Fair | Extreme compression |
| `Q3_K` | ~74% | Good | High compression |
| `Q4_K` | ~67% | Very Good | **Best K-series balance** |
| `Q5_K` | ~59% | Excellent | High quality |
| `Q6_K` | ~51% | Excellent+ | Maximum K quality |

## License

MIT
