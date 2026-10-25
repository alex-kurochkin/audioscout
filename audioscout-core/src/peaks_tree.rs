/// Peaks tree - иерархическая структура для быстрого получения min/max/RMS
///
/// Вместо O(n) для каждого пикселя, дерево позволяет получать данные за O(log n)

const BLOCK: usize = 32;
const FANOUT: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub min: f32,
    pub max: f32,
    pub squares: f32,
}

impl Block {
    pub const EMPTY: Block = Block {
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        squares: 0.0,
    };

    pub fn point(s: f32) -> Self {
        Block {
            min: s,
            max: s,
            squares: if s.is_finite() { s * s } else { 0.0 }
        }
    }

    pub fn of(samples: &[f32]) -> Self {
        samples.iter().fold(Self::EMPTY, |b, &s| b.merge(Self::point(s)))
    }

    pub fn merge(self, o: Block) -> Block {
        Block {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
            squares: self.squares + o.squares,
        }
    }

    pub fn range(self) -> (f32, f32) {
        if self.min > self.max { (0.0, 0.0) } else { (self.min, self.max) }
    }

    pub fn rms(self, count: usize) -> f32 {
        if count == 0 { return 0.0; }
        (self.squares / count as f32).sqrt()
    }
}

pub struct Peaks {
    pub len: usize,
    pub levels: Vec<Vec<Block>>,
}

impl Peaks {
    pub fn build(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Peaks { len: 0, levels: vec![] };
        }
        let mut levels: Vec<Vec<Block>> = vec![
            samples.chunks(BLOCK).map(Block::of).collect()
        ];
        while let Some(last) = levels.last().filter(|l| l.len() > 1) {
            let next = last.chunks(FANOUT)
                .map(|c| c.iter().fold(Block::EMPTY, |a, &b| a.merge(b)))
                .collect();
            levels.push(next);
        }
        Peaks { len: samples.len(), levels }
    }

    pub fn minmax(&self, samples: &[f32], start: usize, end: usize) -> (f32, f32) {
        self.sum(samples, start, end).range()
    }

    pub fn sum(&self, samples: &[f32], start: usize, end: usize) -> Block {
        let end = end.min(self.len).min(samples.len());
        let (first, last) = (start.div_ceil(BLOCK), end / BLOCK);
        let raw = |a: usize, b: usize| Block::of(samples.get(a..b).unwrap_or_default());
        if first >= last { return raw(start, end); }
        raw(start, first * BLOCK)
            .merge(raw(last * BLOCK, end))
            .merge(self.blocks(first, last))
    }

    fn blocks(&self, mut lo: usize, mut hi: usize) -> Block {
        let mut acc = Block::EMPTY;
        for (lvl, level) in self.levels.iter().enumerate() {
            let mut add = |i: usize| {
                acc = acc.merge(level.get(i).copied().unwrap_or(Block::EMPTY));
            };
            if lvl + 1 == self.levels.len() || hi.saturating_sub(lo) < 2 * FANOUT {
                (lo..hi).for_each(add);
                break;
            }
            while lo % FANOUT != 0 { add(lo); lo += 1; }
            while hi % FANOUT != 0 { hi -= 1; add(hi); }
            lo /= FANOUT;
            hi /= FANOUT;
        }
        acc
    }

    /// Получить min/max для каждого столбца изображения
    pub fn spans(&self, samples: &[f32], start: f64, step: f64, cols: usize) -> Vec<(f32, f32)> {
        (0..cols).map(|c| {
            let a = (start + step * c as f64).max(0.0) as usize;
            let b = (start + step * (c + 1) as f64).ceil() as usize;
            self.minmax(samples, a, b)
        }).collect()
    }

    /// Получить RMS для каждого столбца изображения
    pub fn rms_spans(&self, samples: &[f32], start: f64, step: f64, cols: usize) -> Vec<f32> {
        (0..cols).map(|c| {
            let a = (start + step * c as f64).floor() as usize;
            let b = ((start + step * (c + 1) as f64).ceil() as usize)
                .min(self.len).min(samples.len());
            let block = self.sum(samples, a, b);
            block.rms(b.saturating_sub(a))
        }).collect()
    }
}

/// Mixdown interleaved → mono f32
pub fn mixdown_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 { return interleaved.to_vec(); }
    let frames = interleaved.len() / channels;
    let inv = 1.0 / channels as f32;
    (0..frames).map(|f| {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += interleaved[f * channels + ch];
        }
        sum * inv
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_merge() {
        let b1 = Block::point(0.5);
        let b2 = Block::point(-0.3);
        let merged = b1.merge(b2);
        assert_eq!(merged.min, -0.3);
        assert_eq!(merged.max, 0.5);
    }

    #[test]
    fn test_peaks_build() {
        let samples = vec![0.1, 0.2, -0.1, -0.2, 0.3, -0.3, 0.4, -0.4];
        let peaks = Peaks::build(&samples);
        assert_eq!(peaks.len, 8);
        assert!(!peaks.levels.is_empty());
    }

    #[test]
    fn test_mixdown_mono() {
        let interleaved = vec![0.5, 0.5, -0.5, -0.5, 1.0, 1.0];
        let mono = mixdown_to_mono(&interleaved, 2);
        assert_eq!(mono.len(), 3);
        assert_eq!(mono[0], 0.5);
        assert_eq!(mono[1], -0.5);
        assert_eq!(mono[2], 1.0);
    }
}
