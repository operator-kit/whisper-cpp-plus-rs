use std::ffi::CString;
use whisper_cpp_plus_sys as ffi;

#[derive(Clone, Copy, Debug)]
pub enum SamplingStrategy {
    Greedy { best_of: i32 },
    BeamSearch { beam_size: i32 },
}

#[derive(Clone)]
pub struct FullParams {
    pub(crate) inner: ffi::whisper_full_params,
    language: Option<CString>,
    initial_prompt: Option<CString>,
}

// FullParams is Send and Sync because we only use it in controlled contexts
unsafe impl Send for FullParams {}
unsafe impl Sync for FullParams {}

impl FullParams {
    pub fn new(strategy: SamplingStrategy) -> Self {
        let inner = unsafe {
            match strategy {
                SamplingStrategy::Greedy { best_of } => {
                    let mut params = ffi::whisper_full_default_params(
                        ffi::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY,
                    );
                    params.greedy.best_of = best_of;
                    params
                }
                SamplingStrategy::BeamSearch { beam_size } => {
                    let mut params = ffi::whisper_full_default_params(
                        ffi::whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH,
                    );
                    params.beam_search.beam_size = beam_size;
                    params
                }
            }
        };

        let mut params = Self {
            inner,
            language: None,
            initial_prompt: None,
        };

        params.inner.n_threads = (num_cpus::get() / 2).max(1) as i32;
        params.inner.suppress_blank = true;
        params.inner.suppress_nst = true;
        params.inner.temperature = 0.0;
        params.inner.max_initial_ts = 1.0;
        params.inner.length_penalty = -1.0;

        params
    }

    pub(crate) fn as_raw(&self) -> ffi::whisper_full_params {
        let mut params = self.inner;

        if let Some(ref lang) = self.language {
            params.language = lang.as_ptr();
        }

        if let Some(ref prompt) = self.initial_prompt {
            params.initial_prompt = prompt.as_ptr();
        }

        params
    }

    pub fn language(mut self, lang: &str) -> Self {
        self.language = CString::new(lang).ok();
        if let Some(ref lang_cstr) = self.language {
            self.inner.language = lang_cstr.as_ptr();
        }
        self
    }

    pub fn translate(mut self, translate: bool) -> Self {
        self.inner.translate = translate;
        self
    }

    pub fn no_context(mut self, no_context: bool) -> Self {
        self.inner.no_context = no_context;
        self
    }

    pub fn no_timestamps(mut self, no_timestamps: bool) -> Self {
        self.inner.no_timestamps = no_timestamps;
        self
    }

    pub fn single_segment(mut self, single_segment: bool) -> Self {
        self.inner.single_segment = single_segment;
        self
    }

    pub fn print_special(mut self, print_special: bool) -> Self {
        self.inner.print_special = print_special;
        self
    }

    pub fn print_progress(mut self, print_progress: bool) -> Self {
        self.inner.print_progress = print_progress;
        self
    }

    pub fn print_realtime(mut self, print_realtime: bool) -> Self {
        self.inner.print_realtime = print_realtime;
        self
    }

    pub fn print_timestamps(mut self, print_timestamps: bool) -> Self {
        self.inner.print_timestamps = print_timestamps;
        self
    }

    pub fn token_timestamps(mut self, token_timestamps: bool) -> Self {
        self.inner.token_timestamps = token_timestamps;
        self
    }

    pub fn thold_pt(mut self, thold_pt: f32) -> Self {
        self.inner.thold_pt = thold_pt;
        self
    }

    pub fn thold_ptsum(mut self, thold_ptsum: f32) -> Self {
        self.inner.thold_ptsum = thold_ptsum;
        self
    }

    pub fn max_len(mut self, max_len: i32) -> Self {
        self.inner.max_len = max_len;
        self
    }

    pub fn split_on_word(mut self, split_on_word: bool) -> Self {
        self.inner.split_on_word = split_on_word;
        self
    }

    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.inner.max_tokens = max_tokens;
        self
    }


    pub fn debug_mode(mut self, debug_mode: bool) -> Self {
        self.inner.debug_mode = debug_mode;
        self
    }

    pub fn audio_ctx(mut self, audio_ctx: i32) -> Self {
        self.inner.audio_ctx = audio_ctx;
        self
    }

    pub fn tdrz_enable(mut self, tdrz_enable: bool) -> Self {
        self.inner.tdrz_enable = tdrz_enable;
        self
    }

    pub fn suppress_regex(mut self, suppress_regex: Option<&str>) -> Self {
        if let Some(regex) = suppress_regex {
            if let Ok(c_regex) = CString::new(regex) {
                self.inner.suppress_regex = c_regex.as_ptr();
            }
        } else {
            self.inner.suppress_regex = std::ptr::null();
        }
        self
    }

    pub fn initial_prompt(mut self, prompt: &str) -> Self {
        self.initial_prompt = CString::new(prompt).ok();
        if let Some(ref prompt_cstr) = self.initial_prompt {
            self.inner.initial_prompt = prompt_cstr.as_ptr();
        }
        self
    }

    pub fn prompt_tokens(mut self, tokens: &[i32]) -> Self {
        self.inner.prompt_tokens = tokens.as_ptr();
        self.inner.prompt_n_tokens = tokens.len() as i32;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.inner.temperature = temperature;
        self
    }

    pub fn temperature_inc(mut self, temperature_inc: f32) -> Self {
        self.inner.temperature_inc = temperature_inc;
        self
    }

    pub fn entropy_thold(mut self, entropy_thold: f32) -> Self {
        self.inner.entropy_thold = entropy_thold;
        self
    }

    pub fn logprob_thold(mut self, logprob_thold: f32) -> Self {
        self.inner.logprob_thold = logprob_thold;
        self
    }

    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.inner.n_threads = n_threads;
        self
    }

    pub fn offset_ms(mut self, offset_ms: i32) -> Self {
        self.inner.offset_ms = offset_ms;
        self
    }

    pub fn duration_ms(mut self, duration_ms: i32) -> Self {
        self.inner.duration_ms = duration_ms;
        self
    }
}

impl Default for FullParams {
    fn default() -> Self {
        Self::new(SamplingStrategy::Greedy { best_of: 1 })
    }
}

#[derive(Clone)]
pub struct TranscriptionParams {
    params: FullParams,
}

impl TranscriptionParams {
    pub fn builder() -> TranscriptionParamsBuilder {
        TranscriptionParamsBuilder::new()
    }

    pub(crate) fn into_full_params(self) -> FullParams {
        self.params
    }
}

#[derive(Clone)]
pub struct TranscriptionParamsBuilder {
    params: FullParams,
}

impl TranscriptionParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: FullParams::default(),
        }
    }

    pub fn language(mut self, lang: &str) -> Self {
        self.params = self.params.language(lang);
        self
    }

    pub fn translate(mut self, translate: bool) -> Self {
        self.params = self.params.translate(translate);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.params = self.params.temperature(temperature);
        self
    }

    pub fn enable_timestamps(mut self) -> Self {
        self.params = self.params.no_timestamps(false);
        self
    }

    pub fn disable_timestamps(mut self) -> Self {
        self.params = self.params.no_timestamps(true);
        self
    }

    pub fn single_segment(mut self, single: bool) -> Self {
        self.params = self.params.single_segment(single);
        self
    }

    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.params = self.params.max_tokens(max_tokens);
        self
    }

    pub fn initial_prompt(mut self, prompt: &str) -> Self {
        self.params = self.params.initial_prompt(prompt);
        self
    }

    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.params = self.params.n_threads(n_threads);
        self
    }

    pub fn build(self) -> TranscriptionParams {
        TranscriptionParams {
            params: self.params,
        }
    }
}

impl Default for TranscriptionParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}