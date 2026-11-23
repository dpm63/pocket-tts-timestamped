//! Streaming monotone word alignment, ported from this fork's Python timestamps/alignment.py.
//! All offsets are caller-owned text coordinates; times are source PCM samples, never wall time.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Unit {
    pub start: usize,
    pub length: usize,
    pub word_index: i32,
    pub synthetic: bool,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Layout {
    pub units: Vec<Unit>,
    pub token_to_unit: Vec<Vec<f32>>,
}
impl Layout {
    pub fn validate(&self, tokens: usize) -> Result<()> {
        ensure!(
            !self.units.is_empty()
                && self.units.len() <= 4096
                && self.token_to_unit.len() == tokens,
            "invalid timestamp layout dimensions"
        );
        for row in &self.token_to_unit {
            ensure!(
                row.len() == self.units.len() && row.iter().all(|x| x.is_finite() && *x >= 0.),
                "invalid timestamp mapping"
            );
            let sum: f32 = row.iter().sum();
            ensure!(
                sum == 0. || (sum - 1.).abs() < 1e-4,
                "timestamp mapping rows must sum to one or zero"
            );
        }
        let words: Vec<_> = self.units.iter().filter(|u| u.word_index >= 0).collect();
        ensure!(
            words
                .iter()
                .all(|u| u.length > 0 && u.start.checked_add(u.length).is_some()),
            "invalid word span"
        );
        ensure!(
            words.windows(2).all(|w| (w[0].word_index < w[1].word_index
                && w[0].start + w[0].length <= w[1].start)
                || (w[0].word_index == w[1].word_index
                    && w[0].start == w[1].start
                    && w[0].length == w[1].length)),
            "unordered timestamp words"
        );
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordEvent {
    pub start: usize,
    pub length: usize,
    pub word_index: i32,
    pub sample_offset: usize,
    /// False = start; true = end. Clients can highlight on starts without waiting for word ends.
    pub is_end: bool,
}

/// Select only calibrations tied to known source weights, not merely a language/architecture.
/// Unknown checkpoints keep ordinary synthesis and the sentence-level presentation fallback.
pub fn calibrated_heads(
    sha: &str,
    language: &str,
    shape: &crate::config::FlowLMTransformerConfig,
) -> Result<Option<Vec<crate::config::TimestampHead>>> {
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../calibrations.json"))?;
    for entry in entries {
        if entry["sha256"].as_str() == Some(sha)
            && entry["language"].as_str() == Some(language)
            && entry["numLayers"].as_u64() == Some(shape.num_layers as u64)
            && entry["numHeads"].as_u64() == Some(shape.num_heads as u64)
            && entry["modelDim"].as_u64() == Some(shape.d_model as u64)
        {
            let heads: Vec<crate::config::TimestampHead> =
                serde_json::from_value(entry["heads"].clone())?;
            ensure!(
                !heads.is_empty()
                    && heads
                        .iter()
                        .all(|h| h.layer < shape.num_layers && h.head < shape.num_heads),
                "Invalid checkpoint calibration"
            );
            return Ok(Some(heads));
        }
    }
    Ok(None)
}

pub fn is_voiced(pcm: &[f32]) -> bool {
    let energy: f32 = pcm.iter().map(|v| v * v).sum();
    energy > 1e-6 * pcm.len() as f32
}
pub struct Alignment {
    units: Vec<Unit>,
    words: Vec<usize>,
    next: usize,
    open: Option<usize>,
}
impl Alignment {
    pub fn new(units: Vec<Unit>) -> Self {
        let words = units
            .iter()
            .enumerate()
            .filter(|(_, u)| u.word_index >= 0)
            .map(|(i, _)| i)
            .collect();
        Self {
            units,
            words,
            next: 0,
            open: None,
        }
    }
    fn event(&self, pos: usize, sample: usize, is_end: bool) -> WordEvent {
        let u = &self.units[self.words[pos]];
        WordEvent {
            start: u.start,
            length: u.length,
            word_index: u.word_index,
            sample_offset: sample,
            is_end,
        }
    }
    fn close(&mut self, sample: usize) -> WordEvent {
        let position = self.open.take().expect("open word");
        self.event(position, sample, true)
    }
    fn open_next(&mut self, sample: usize) -> WordEvent {
        let position = self.next;
        self.next += 1;
        self.open = Some(position);
        self.event(position, sample, false)
    }
    pub fn process(
        &mut self,
        scores: &[f32],
        voiced: bool,
        sample: usize,
    ) -> Result<Vec<WordEvent>> {
        ensure!(
            scores.len() == self.units.len() && scores.iter().all(|x| x.is_finite()),
            "invalid unit scores"
        );
        let mut events = Vec::new();
        if !voiced {
            if let Some(pos) = self.open {
                let unit = self.words[pos];
                let future = scores[unit + 1..].iter().any(|v| *v > scores[unit]);
                let final_word = pos == self.words.len() - 1;
                let punctuation = self.units[unit + 1..]
                    .iter()
                    .any(|u| u.word_index < 0 && !u.synthetic);
                if future || (final_word && !punctuation) {
                    events.push(self.close(sample));
                }
            }
        } else if let Some(pos) = self.open {
            if self.next < self.words.len() {
                let current = scores[self.words[pos]];
                let next = scores[self.words[self.next]] > current;
                let later = current < 0.001
                    && self.words[self.next + 1..]
                        .iter()
                        .any(|u| scores[*u] > current);
                if next || later {
                    events.push(self.close(sample));
                    events.push(self.open_next(sample));
                }
            }
        } else if self.next < self.words.len() {
            events.push(self.open_next(sample));
        }
        Ok(events)
    }
    pub fn finish(&mut self, sample: usize) -> Vec<WordEvent> {
        let result = if self.open.is_some() {
            vec![self.close(sample)]
        } else {
            vec![]
        };
        self.next = self.words.len();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn units() -> Vec<Unit> {
        vec![
            Unit {
                start: 0,
                length: 4,
                word_index: 0,
                synthetic: false,
            },
            Unit {
                start: 5,
                length: 5,
                word_index: 1,
                synthetic: false,
            },
            Unit {
                start: 10,
                length: 1,
                word_index: -1,
                synthetic: false,
            },
        ]
    }
    #[test]
    fn streaming_starts_precede_audio_and_punctuation_closes_final_word() {
        let mut a = Alignment::new(units());
        assert!(a.process(&[1., 0., 0.], false, 0).unwrap().is_empty());
        assert_eq!(
            a.process(&[1., 0., 0.], true, 1920).unwrap()[0],
            WordEvent {
                start: 0,
                length: 4,
                word_index: 0,
                sample_offset: 1920,
                is_end: false
            }
        );
        let e = a.process(&[0.1, 0.8, 0.1], true, 3840).unwrap();
        assert!(e[0].is_end && !e[1].is_end && e[1].word_index == 1);
        assert!(a.process(&[0., 0.9, 0.1], false, 5760).unwrap().is_empty());
        assert!(a.process(&[0., 0.1, 0.9], false, 7680).unwrap()[0].is_end);
        assert!(a.finish(9600).is_empty());
    }
    #[test]
    fn finish_closes_open_word_once_and_does_not_invent_skipped_words() {
        let mut a = Alignment::new(units());
        a.process(&[1., 0., 0.], true, 0).unwrap();
        assert_eq!(a.finish(1920).len(), 1);
        assert!(a.finish(1920).is_empty());
        assert!(a.process(&[0., 1., 0.], true, 3840).unwrap().is_empty());
    }
    #[test]
    fn invalid_mapping_and_scores_are_errors() {
        let l = Layout {
            units: units(),
            token_to_unit: vec![vec![1., 0., 0.]],
        };
        assert!(l.validate(1).is_ok());
        assert!(l.validate(2).is_err());
        assert!(
            Alignment::new(units())
                .process(&[f32::NAN, 0., 0.], true, 0)
                .is_err()
        );
        assert!(!is_voiced(&[]));
        assert!(!is_voiced(&[0.; 1920]));
        assert!(is_voiced(&[0.1; 1920]));
    }
}

#[cfg(test)]
mod capture_tests {
    use ptts::timestamp_attention::AttentionCapture;
    use xn::CpuDevice;
    use xn::{Tensor, TensorView};
    #[test]
    fn selected_heads_reduce_only_text_keys_and_require_every_head() {
        let q = Tensor::from_vec(vec![1f32, 0., 0., 1.], (1, 1, 2, 2), &CpuDevice).unwrap();
        let k = Tensor::from_vec(
            vec![100f32, 100., 100., 100., 1., 0., 0., 1., 0., 1., 1., 0.],
            (1, 3, 2, 2),
            &CpuDevice,
        )
        .unwrap();
        let mut capture =
            AttentionCapture::new(vec![(3, 0), (4, 1)], 1, vec![vec![1., 0.], vec![0., 1.]])
                .unwrap();
        capture.begin_frame();
        capture
            .capture(3, &q, &TensorView::from(k.clone()))
            .unwrap();
        assert!(capture.finish_frame().is_err());
        capture.capture(4, &q, &TensorView::from(k)).unwrap();
        let scores = capture.finish_frame().unwrap();
        let expected = 1. / (1. + (-1f32 / 2f32.sqrt()).exp());
        assert!((scores[0] - expected).abs() < 1e-6);
        assert!((scores.iter().sum::<f32>() - 1.).abs() < 1e-6);
        capture.begin_frame();
        assert!(capture.finish_frame().is_err());
    }
    #[test]
    fn invalid_heads_and_mapping_do_not_silently_produce_timestamps() {
        assert!(AttentionCapture::new(vec![], 0, vec![vec![1.]]).is_err());
        assert!(AttentionCapture::new(vec![(0, 0), (0, 0)], 0, vec![vec![1.]]).is_err());
        assert!(AttentionCapture::new(vec![(0, 0)], 0, vec![vec![f32::NAN]]).is_err());
    }
}

#[cfg(test)]
mod calibration_tests {
    #[test]
    fn legacy_and_october_italian_use_their_own_heads_and_unknown_weights_do_not_guess() {
        let (config, _) = crate::config::Config::load(Some("italian"), None).unwrap();
        let shape = &config.flow_lm.transformer;
        let old = super::calibrated_heads(
            "6af958c43372082b91af79d98d07d3fa3c2a2e41b628e5b119fdb46e19f0e49f",
            "it",
            shape,
        )
        .unwrap()
        .unwrap();
        let new = super::calibrated_heads(
            "16365c43eaabee9b222cd414d6a48f0976b4800a21fe5d55253f645eff7eeb1f",
            "it",
            shape,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            old.iter().map(|h| (h.layer, h.head)).collect::<Vec<_>>(),
            vec![(3, 9), (4, 5), (3, 14), (3, 0)]
        );
        assert_eq!(
            new.iter().map(|h| (h.layer, h.head)).collect::<Vec<_>>(),
            vec![(3, 9), (4, 5), (3, 10)]
        );
        assert!(
            super::calibrated_heads("unknown", "it", shape)
                .unwrap()
                .is_none()
        );
        assert!(
            super::calibrated_heads(
                "6af958c43372082b91af79d98d07d3fa3c2a2e41b628e5b119fdb46e19f0e49f",
                "en",
                shape
            )
            .unwrap()
            .is_none()
        );
    }
}
