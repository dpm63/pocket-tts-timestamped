//! Runs fixtures produced by the Python implementation through the Rust aligner.
use anyhow::{Context, Result, ensure};
use pocket_tts::timestamps::{Alignment, Layout, WordEvent, is_voiced};
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
#[derive(Deserialize)]
struct VoicedFrame {
    pcm: Vec<f32>,
    voiced: bool,
}
#[derive(Deserialize)]
struct Fixtures {
    alignment: Vec<Case>,
    voiced: Vec<VoicedFrame>,
}
fn main() -> Result<()> {
    let file = std::env::args().nth(1).context("fixture path required")?;
    let fixtures: Fixtures = serde_json::from_slice(&std::fs::read(file)?)?;
    for case in &fixtures.alignment {
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
    for (index, frame) in fixtures.voiced.iter().enumerate() {
        ensure!(
            is_voiced(&frame.pcm) == frame.voiced,
            "Python/Rust voicing differs in frame {index}: expected voiced={}",
            frame.voiced
        );
    }
    println!(
        "Python/Rust alignment parity: {} cases, {} voicing frames",
        fixtures.alignment.len(),
        fixtures.voiced.len()
    );
    Ok(())
}
