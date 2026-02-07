use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whisper_cpp_plus::WhisperContext;

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // Placeholder benchmark - will be replaced in Phase 5
            black_box(42)
        });
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);