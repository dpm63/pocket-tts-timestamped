//! Runs fixtures produced by the Python implementation through the Rust aligner.
use anyhow::{Context, Result, ensure};
use pocket_tts::timestamps::{Alignment, Layout, WordEvent};
use serde::Deserialize;
#[derive(Deserialize)]
struct Frame {
    scores: Vec<f32>,
    voiced: bool,
    sample: usize,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    layout: Layout,
    frames: Vec<Frame>,
    end: usize,
    expected: Vec<WordEvent>,
}
fn main() -> Result<()> {
    let file = std::env::args().nth(1).context("fixture path required")?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(file)?)?;
    for case in &cases {
        case.layout.validate(case.layout.token_to_unit.len())?;
        let mut align = Alignment::new(case.layout.units.clone());
        let mut actual = Vec::new();
        for f in &case.frames {
            actual.extend(align.process(&f.scores, f.voiced, f.sample)?);
        }
        actual.extend(align.finish(case.end));
        ensure!(
            actual == case.expected,
            "Python/Rust alignment differs in {}: actual={actual:?}, expected={:?}",
            case.name,
            case.expected
        );
    }
    println!("Python/Rust alignment parity: {} cases", cases.len());
    Ok(())
}
