use biquad::{Biquad, Coefficients, DirectForm1, Hertz, ToHertz, Q_BUTTERWORTH_F32};
use nih_plug::{prelude::*, util::MINUS_INFINITY_DB};
use nih_plug_egui::{
    create_egui_editor,
    egui::{self, CollapsingHeader, ScrollArea},
    widgets, EguiState,
};
use realfft::{
    num_complex::{Complex, Complex32},
    num_traits::Zero,
    ComplexToReal, RealFftPlanner, RealToComplex,
};
use std::{
    f32::consts::PI,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
    usize,
};

/// The size of the windows we'll process at a time.
const WINDOW_SIZE: usize = 16384;
/// The length of the FFT window we will use to perform FFT convolution. This includes padding to
/// prevent time domain aliasing as a result of cyclic convolution.
const FFT_WINDOW_SIZE: usize = WINDOW_SIZE;

const MAX_MAGNITUDE_THRESHOLD: f32 = 0.05;

pub struct Ge79Plugin {
    params: Arc<Ge79PluginParams>,

    /// An adapter that performs most of the overlap-add algorithm for us.
    stft: util::StftHelper,

    /// The algorithm for the FFT operation.
    r2c_plan: Arc<dyn RealToComplex<f32>>,
    /// The algorithm for the IFFT operation.
    c2r_plan: Arc<dyn ComplexToReal<f32>>,
    /// The output of our real->complex FFT.
    complex_fft_buffer: Vec<Complex32>,
}

impl Default for Ge79Plugin {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        let r2c_plan = planner.plan_fft_forward(FFT_WINDOW_SIZE);
        let c2r_plan = planner.plan_fft_inverse(FFT_WINDOW_SIZE);
        let mut real_fft_buffer = r2c_plan.make_input_vec();
        let mut complex_fft_buffer = r2c_plan.make_output_vec();

        // RustFFT doesn't actually need a scratch buffer here, so we'll pass an empty buffer
        // instead
        r2c_plan
            .process_with_scratch(&mut real_fft_buffer, &mut complex_fft_buffer, &mut [])
            .unwrap();

        Self {
            params: Arc::new(Ge79PluginParams::default()),

            // We'll process the input in `WINDOW_SIZE` chunks, but our FFT window is slightly
            // larger to account for time domain aliasing so we'll need to add some padding ot each
            // block.
            stft: util::StftHelper::new(2, WINDOW_SIZE, FFT_WINDOW_SIZE - WINDOW_SIZE),

            r2c_plan,
            c2r_plan,
            complex_fft_buffer,
        }
    }
}

#[derive(Debug, Params)]
struct Ge79PluginParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    #[nested(array, group = "Channels")]
    pub channels: Vec<ChannelParams>,

    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
}

impl Default for Ge79PluginParams {
    fn default() -> Self {
        let mut channels = Vec::with_capacity(16);
        for _ in 0..16 {
            channels.push(ChannelParams::default());
        }
        Self {
            channels,
            editor_state: EguiState::from_size(300, 180),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Enum)]
enum ChannelSelection {
    Left,
    Right,
}

#[derive(Debug)]
struct ChannelState {
    magnitude_filter: DirectForm1<f32>,
    frequency_filter: DirectForm1<f32>,
    phase_filter: DirectForm1<f32>,
    sample_rate: f32,
    last_phase: f32,
    last_frequency: f32,
    last_second_start: Instant,
    current_sample_count: u32,
    last_second_sample_count: u32,
}

impl ChannelState {
    fn new(fs: Hertz<f32>) -> Result<Self, biquad::Errors> {
        let filter = DirectForm1::<f32>::new(Self::create_filter_coefficients(fs, 1.hz())?);
        Ok(Self {
            magnitude_filter: filter,
            frequency_filter: filter,
            phase_filter: filter,
            sample_rate: fs.hz(),
            last_phase: 0.0,
            last_frequency: 0.0,
            last_second_start: Instant::now(),
            current_sample_count: 0,
            last_second_sample_count: 0,
        })
    }

    fn create_filter_coefficients(
        fs: Hertz<f32>,
        f0: Hertz<f32>,
    ) -> Result<Coefficients<f32>, biquad::Errors> {
        Coefficients::<f32>::from_params(biquad::Type::LowPass, fs, f0, Q_BUTTERWORTH_F32)
    }

    fn reset(&mut self) {
        self.magnitude_filter.reset_state();
        self.frequency_filter.reset_state();
        self.phase_filter.reset_state();
        self.last_phase = 0.0;
        self.last_frequency = 0.0;
    }

    fn update(
        &mut self,
        magnitude: f32,
        phase: f32,
        frequency: f32,
        fs: Hertz<f32>,
    ) -> Result<(f32, f32, f32), biquad::Errors> {
        // nih_log!("{magnitude}, {phase}, {frequency}, {fs:?}");
        if self.sample_rate != fs.hz() {
            nih_log!("update coefficients");
            self.magnitude_filter
                .update_coefficients(Self::create_filter_coefficients(fs, 0.1.hz())?);
            self.phase_filter
                .update_coefficients(Self::create_filter_coefficients(fs, 0.1.hz())?);
            self.frequency_filter
                .update_coefficients(Self::create_filter_coefficients(fs, 0.2.hz())?);
            self.sample_rate = fs.hz();
        }
        if magnitude > MAX_MAGNITUDE_THRESHOLD {
            let frequency = self.frequency_filter.run(frequency);
            let phase = self.phase_filter.run(phase);
            self.last_frequency = frequency;
            self.last_phase = phase;

            Ok((self.magnitude_filter.run(magnitude), phase, frequency))
        } else {
            Ok((
                self.magnitude_filter.run(magnitude),
                self.last_phase,
                self.last_frequency,
            ))
        }
    }
}

#[derive(Debug, Params)]
struct ChannelParams {
    #[id = "volume"]
    volume: FloatParam,
    #[id = "chan"]
    channel: EnumParam<ChannelSelection>,
    #[id = "otnum"]
    overtone_num: IntParam,
    #[id = "phase"]
    phase_shift: FloatParam,
    #[id = "invpha"]
    invert_phase: BoolParam,
    state: Mutex<ChannelState>,
}

impl Default for ChannelParams {
    fn default() -> Self {
        Self {
            volume: FloatParam::new(
                "Volume",
                util::db_to_gain(MINUS_INFINITY_DB + 1.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(MINUS_INFINITY_DB + 1.0),
                    max: util::db_to_gain(-0.1),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(MINUS_INFINITY_DB + 1.0, -0.1),
                },
            )
            // Because the gain parameter is stored as linear gain instead of storing the value as
            // decibels, we need logarithmic smoothing
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            // There are many predefined formatters we can use here. If the gain was stored as
            // decibels instead of as a linear gain value, we could have also used the
            // `.with_step_size(0.1)` function to get internal rounding.
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            channel: EnumParam::new("Channel Selection", ChannelSelection::Left),
            overtone_num: IntParam::new("Overtone", 1, IntRange::Linear { min: 1, max: 16 }),
            phase_shift: FloatParam::new(
                "Phase Shift",
                0.0,
                FloatRange::Linear { min: -PI, max: PI },
            ),
            invert_phase: BoolParam::new("Invert Phase", false),
            state: Mutex::new(
                ChannelState::new(44.1.khz()).expect("Could not create channel state"),
            ),
        }
    }
}

impl Plugin for Ge79Plugin {
    const NAME: &'static str = "GE 79";
    const VENDOR: &'static str = "DerFetzer";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "kontakt@der-fetzer.de";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        _buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        // The plugin's latency consists of the block size from the overlap-add procedure and half
        // of the filter kernel's size (since we're using a linear phase/symmetrical convolution
        // kernel)
        context.set_latency_samples(self.stft.latency_samples());

        true
    }

    fn reset(&mut self) {
        // Normally we'd also initialize the STFT helper for the correct channel count here, but we
        // only do stereo so that's not necessary. Setting the block size also zeroes out the
        // buffers.
        self.stft.set_block_size(WINDOW_SIZE);
        for channel in &self.params.channels {
            channel.state.lock().unwrap().reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.stft
            .process_overlap_add(buffer, 1, |channel_idx, real_fft_buffer| {
                // Forward FFT, `real_fft_buffer` is already padded with zeroes, and the
                // padding from the last iteration will have already been added back to the start of
                // the buffer
                self.r2c_plan
                    .process_with_scratch(real_fft_buffer, &mut self.complex_fft_buffer, &mut [])
                    .unwrap();

                let (max_idx, max) = self
                    .complex_fft_buffer
                    .iter()
                    .enumerate()
                    .max_by(|(_, x), (_, y)| {
                        x.norm()
                            .partial_cmp(&y.norm())
                            .expect("Tried to compare NaN")
                    })
                    .expect("Empty buffer");
                let max = *max;

                for value in &mut self.complex_fft_buffer {
                    *value = Complex::zero();
                }

                let sample_rate = context.transport().sample_rate;
                let (max_magnitude, max_phase) = max.to_polar();
                let max_freq = max_idx as f32 * sample_rate / FFT_WINDOW_SIZE as f32;

                for channel in &self.params.channels {
                    match channel.channel.value() {
                        ChannelSelection::Left if channel_idx != 0 => continue,
                        ChannelSelection::Right if channel_idx != 1 => continue,
                        _ => (),
                    };
                    let mut channel_state = channel.state.lock().unwrap();
                    if channel_state.last_second_start.elapsed() >= Duration::from_secs(1) {
                        channel_state.last_second_start = Instant::now();
                        channel_state.last_second_sample_count = channel_state.current_sample_count;
                        channel_state.current_sample_count = 0;
                    } else {
                        channel_state.current_sample_count += 1;
                    }

                    if channel.volume.value() > util::db_to_gain(MINUS_INFINITY_DB + 1.0) {
                        let (magnitude, phase, frequency) = match channel_state.update(
                            if max_magnitude > MAX_MAGNITUDE_THRESHOLD {
                                channel.volume.value()
                            } else {
                                0.0
                            },
                            max_phase
                                + channel.phase_shift.value()
                                + if channel.invert_phase.value() {
                                    PI
                                } else {
                                    0.0
                                },
                            max_freq * channel.overtone_num.value() as f32,
                            (context.transport().sample_rate / FFT_WINDOW_SIZE as f32).hz(), // *2 probably because of overlap_times = 1?!
                        ) {
                            Ok(res) => res,
                            Err(err) => {
                                nih_log!("{err:?}");
                                continue;
                            }
                        };
                        let index = (frequency * FFT_WINDOW_SIZE as f32 / sample_rate) as usize;
                        if index == 0 || index == self.complex_fft_buffer.len() - 1 {
                            continue;
                        }
                        nih_log!("index: {index}");
                        if let Some(bin) = self.complex_fft_buffer.get_mut(index) {
                            let complex = Complex::from_polar(magnitude, phase);
                            *bin += complex;
                        }
                    } else {
                        channel_state.reset();
                    }
                }

                // Inverse FFT back into the scratch buffer. This will be added to a ring buffer
                // which gets written back to the host at a one block delay.
                self.c2r_plan
                    .process_with_scratch(&mut self.complex_fft_buffer, real_fft_buffer, &mut [])
                    .unwrap();
            });

        ProcessStatus::Normal
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    ScrollArea::vertical().show(ui, |ui| {
                        for (idx, channel) in params.channels.iter().enumerate() {
                            ui.label("Volume");
                            ui.add(widgets::ParamSlider::for_param(&channel.volume, setter));
                            ui.label("Overtone Number");
                            ui.add(widgets::ParamSlider::for_param(
                                &channel.overtone_num,
                                setter,
                            ));
                            ui.label("Phase Shift");
                            ui.add(widgets::ParamSlider::for_param(
                                &channel.phase_shift,
                                setter,
                            ));
                            CollapsingHeader::new("Debug")
                                .id_source(idx)
                                .show(ui, |ui| {
                                    ui.label(format!(
                                        "sample_rate: {}",
                                        channel.state.lock().unwrap().sample_rate
                                    ));
                                    ui.label(format!(
                                        "last_phase: {}",
                                        channel.state.lock().unwrap().last_phase
                                    ));
                                    ui.label(format!(
                                        "last_frequency: {}",
                                        channel.state.lock().unwrap().last_frequency
                                    ));
                                    ui.label(format!(
                                        "Samples/s: {}",
                                        channel.state.lock().unwrap().last_second_sample_count
                                    ));
                                });
                            ui.separator();
                        }
                    });
                });
            },
        )
    }
}

impl ClapPlugin for Ge79Plugin {
    const CLAP_ID: &'static str = "de.der-fetzer.ge79-plugin";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("A plugin recreating the GE 79 overtone synthesizer");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for Ge79Plugin {
    const VST3_CLASS_ID: [u8; 16] = *b"GE79DerFetzer___";

    // And also don't forget to change these categories
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Filter, Vst3SubCategory::Filter];
}

nih_export_clap!(Ge79Plugin);
nih_export_vst3!(Ge79Plugin);
