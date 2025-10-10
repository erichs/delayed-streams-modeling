// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use candle::{Device, Tensor};
use clap::Parser;
use std::fs::{File, OpenOptions};
use std::io::Write as IoWrite;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// Buffered file writer that flushes based on size or time thresholds.
/// This ensures that file watchers can see updates without excessive delay.
/// Closes and reopens the file on each flush to reliably trigger file system events.
struct BufferedFileWriter {
    file: File,
    path: String,
    bytes_written: usize,
    last_flush: Instant,
    flush_size_threshold: usize,
    flush_time_threshold: Duration,
}

impl BufferedFileWriter {
    fn new(file: File, path: String) -> Self {
        Self {
            file,
            path,
            bytes_written: 0,
            last_flush: Instant::now(),
            flush_size_threshold: 200,  // 200 bytes
            flush_time_threshold: Duration::from_secs(2),  // 2 seconds
        }
    }

    fn write(&mut self, data: &str) -> Result<()> {
        self.file.write_all(data.as_bytes())?;
        self.bytes_written += data.len();

        // Check if we should flush
        let should_flush = self.bytes_written >= self.flush_size_threshold
            || self.last_flush.elapsed() >= self.flush_time_threshold;

        if should_flush {
            self.flush()?;
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // Flush and sync to disk
        self.file.flush()?;
        self.file.sync_all()?;

        // Close and reopen the file to trigger file system events
        // This is necessary for file watchers (like chokidar with fsevents) to detect changes
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        self.bytes_written = 0;
        self.last_flush = Instant::now();
        Ok(())
    }
}

impl Drop for BufferedFileWriter {
    fn drop(&mut self) {
        // Ensure final flush on drop
        let _ = self.flush();
    }
}

#[derive(Debug, Parser)]
struct Args {
    /// The audio input file, in wav/mp3/ogg/... format. Use "--mic" to capture from microphone.
    #[arg(default_value = "--mic")]
    in_file: String,

    /// The repo where to get the model from.
    #[arg(long, default_value = "kyutai/stt-1b-en_fr-candle")]
    hf_repo: String,

    /// Path to the model file in the repo.
    #[arg(long, default_value = "model.safetensors")]
    model_path: String,

    /// Run the model on cpu.
    #[arg(long)]
    cpu: bool,

    /// Display word level timestamps.
    #[arg(long)]
    timestamps: bool,

    /// Display the level of voice activity detection (VAD).
    #[arg(long)]
    vad: bool,

    /// Use microphone input instead of file.
    #[arg(long)]
    mic: bool,

    /// List available audio input devices.
    #[arg(long)]
    list_devices: bool,

    /// Duration in seconds to record from microphone (default: continuous until Ctrl+C).
    #[arg(long)]
    duration: Option<f32>,

    /// Output file path to append transcript to (optional).
    #[arg(long)]
    output: Option<String>,

    /// Audio gain/sensitivity multiplier (1.0 = normal, 2.0 = double volume, etc.).
    /// Useful for picking up softer speech. Recommended range: 1.0-5.0.
    #[arg(long, default_value = "1.0")]
    gain: f32,
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self, vad: bool) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        let extra_heads = if vad {
            Some(moshi::lm::ExtraHeadsConfig {
                num_heads: 4,
                dim: 6,
            })
        } else {
            None
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

struct Model {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    timestamps: bool,
    vad: bool,
    config: Config,
    dev: Device,
}

impl Model {
    fn load_from_hf(args: &Args, dev: &Device) -> Result<Self> {
        // Retrieve the model files from the Hugging Face Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(args.hf_repo.to_string());
        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get(&args.model_path)?;
        let mimi_file = repo.get(&config.mimi_name)?;
        let is_quantized = model_file.to_str().unwrap().ends_with(".gguf");

        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;

        let lm = if is_quantized {
            let vb_lm = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &model_file,
                dev,
            )?;
            moshi::lm::LmModel::new(
                &config.model_config(args.vad),
                moshi::nn::MaybeQuantizedVarBuilder::Quantized(vb_lm),
            )?
        } else {
            let dtype = dev.bf16_default_to_f32();
            let vb_lm = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, dev)?
            };
            moshi::lm::LmModel::new(
                &config.model_config(args.vad),
                moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
            )?
        };

        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), dev)?;
        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;
        Ok(Model {
            state,
            config,
            text_tokenizer,
            timestamps: args.timestamps,
            vad: args.vad,
            dev: dev.clone(),
        })
    }

    fn run(&mut self, mut pcm: Vec<f32>, output_file: Option<String>) -> Result<()> {
        // Add the silence prefix to the audio.
        if self.config.stt_config.audio_silence_prefix_seconds > 0.0 {
            let silence_len =
                (self.config.stt_config.audio_silence_prefix_seconds * 24000.0) as usize;
            pcm.splice(0..0, vec![0.0; silence_len]);
        }
        // Add some silence at the end to ensure all the audio is processed.
        let suffix = (self.config.stt_config.audio_delay_seconds * 24000.0) as usize;
        pcm.resize(pcm.len() + suffix + 24000, 0.0);

        self.process_audio_chunks(&pcm, output_file)?;
        Ok(())
    }

    fn run_streaming(&mut self, audio_receiver: Receiver<Vec<f32>>, output_file: Option<String>) -> Result<()> {
        println!("Starting real-time transcription from microphone...");
        println!("Speak now - transcription will appear as you talk.");
        println!("Press Ctrl+C to stop.");
        if let Some(ref path) = output_file {
            println!("Transcript will be appended to: {}", path);
            println!("(File will be synced every 200 bytes or 2 seconds for file watchers)");
        }
        println!();

        let mut last_print_was_vad = false;
        let mut file_writer = if let Some(ref path) = output_file {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            Some(BufferedFileWriter::new(file, path.clone()))
        } else {
            None
        };

        loop {
            // Receive audio chunk from microphone (blocking)
            let chunk = match audio_receiver.recv() {
                Ok(chunk) => chunk,
                Err(_) => break, // Channel closed, microphone stopped
            };

            // Process the chunk through the model
            let pcm = Tensor::new(&chunk[..], &self.dev)?.reshape((1, 1, ()))?;
            let asr_msgs = self.state.step_pcm(pcm, None, &().into(), |_, _, _| ())?;

            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        if self.vad && prs[2][0] > 0.5 && !last_print_was_vad {
                            let msg = " [end of turn detected]";
                            print!("{}", msg);
                            std::io::stdout().flush()?;
                            if let Some(ref mut writer) = file_writer {
                                writer.write(msg)?;
                            }
                            last_print_was_vad = true;
                        }
                    }
                    moshi::asr::AsrMsg::Word { tokens, .. } => {
                        let word = self
                            .text_tokenizer
                            .decode_piece_ids(tokens)
                            .unwrap_or_else(|_| String::new());
                        // Add space before word for proper spacing
                        let output = format!(" {}", word);
                        print!("{}", output);
                        std::io::stdout().flush()?;
                        if let Some(ref mut writer) = file_writer {
                            writer.write(&output)?;
                        }
                        last_print_was_vad = false;
                    }
                    _ => {}
                }
            }
        }

        println!();
        if let Some(ref mut writer) = file_writer {
            writer.write("\n")?;
            writer.flush()?;
        }
        println!("Transcription stopped.");
        Ok(())
    }

    fn process_audio_chunks(&mut self, pcm: &[f32], output_file: Option<String>) -> Result<()> {
        let mut file_writer = if let Some(ref path) = output_file {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            Some(BufferedFileWriter::new(file, path.clone()))
        } else {
            None
        };

        let mut last_word = None;
        let mut printed_eot = false;
        for pcm_chunk in pcm.chunks(1920) {
            let pcm_tensor = Tensor::new(pcm_chunk, &self.dev)?.reshape((1, 1, ()))?;
            let asr_msgs = self.state.step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;
            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        if self.vad && prs[2][0] > 0.5 && !printed_eot {
                            printed_eot = true;
                            if !self.timestamps {
                                let msg = format!(" <endofturn pr={}>", prs[2][0]);
                                print!("{}", msg);
                                if let Some(ref mut writer) = file_writer {
                                    writer.write(&msg)?;
                                }
                            } else {
                                let msg = format!("<endofturn pr={}>\n", prs[2][0]);
                                print!("{}", msg);
                                if let Some(ref mut writer) = file_writer {
                                    writer.write(&msg)?;
                                }
                            }
                        }
                    }
                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                        printed_eot = false;
                        #[allow(clippy::collapsible_if)]
                        if self.timestamps {
                            if let Some((word, start_time)) = last_word.take() {
                                let msg = format!("[{start_time:5.2}-{stop_time:5.2}] {word}\n");
                                print!("{}", msg);
                                if let Some(ref mut writer) = file_writer {
                                    writer.write(&msg)?;
                                }
                            }
                        }
                    }
                    moshi::asr::AsrMsg::Word {
                        tokens, start_time, ..
                    } => {
                        printed_eot = false;
                        let word = self
                            .text_tokenizer
                            .decode_piece_ids(tokens)
                            .unwrap_or_else(|_| String::new());
                        if !self.timestamps {
                            let output = format!(" {}", word);
                            print!("{}", output);
                            std::io::stdout().flush()?;
                            if let Some(ref mut writer) = file_writer {
                                writer.write(&output)?;
                            }
                        } else {
                            if let Some((word, prev_start_time)) = last_word.take() {
                                let msg = format!("[{prev_start_time:5.2}-{start_time:5.2}] {word}\n");
                                print!("{}", msg);
                                if let Some(ref mut writer) = file_writer {
                                    writer.write(&msg)?;
                                }
                            }
                            last_word = Some((word, *start_time));
                        }
                    }
                }
            }
        }
        if let Some((word, start_time)) = last_word.take() {
            let msg = format!("[{start_time:5.2}-     ] {word}\n");
            print!("{}", msg);
            if let Some(ref mut writer) = file_writer {
                writer.write(&msg)?;
            }
        }
        println!();
        if let Some(ref mut writer) = file_writer {
            writer.flush()?;
        }
        Ok(())
    }
}

fn list_audio_devices() -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    println!("Available audio input devices:");
    println!();

    for (idx, device) in host.input_devices()?.enumerate() {
        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        println!("  [{}] {}", idx, name);

        if let Ok(config) = device.default_input_config() {
            println!("      Sample rate: {} Hz", config.sample_rate().0);
            println!("      Channels: {}", config.channels());
        }
        println!();
    }

    Ok(())
}

fn stream_from_microphone(sender: Sender<Vec<f32>>, gain: f32) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;

    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    println!("Using input device: {}", device.name()?);

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    println!("Recording at {} Hz with {} channel(s)", sample_rate, channels);
    if gain != 1.0 {
        println!("Audio gain: {:.1}x", gain);
    }

    // We need to resample if not 24kHz
    let needs_resampling = sample_rate != 24000;
    if needs_resampling {
        println!("Will resample from {} Hz to 24000 Hz", sample_rate);
    }

    let chunk_size = 1920; // 80ms at 24kHz
    let input_chunk_size = if needs_resampling {
        (chunk_size as f32 * sample_rate as f32 / 24000.0) as usize
    } else {
        chunk_size
    };

    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let audio_buffer_clone = audio_buffer.clone();

    let err_fn = |err| eprintln!("Error in audio stream: {}", err);

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buffer = audio_buffer_clone.lock().unwrap();
                    // Convert to mono if needed and apply gain
                    if channels == 1 {
                        buffer.extend(data.iter().map(|&s| s * gain));
                    } else {
                        for chunk in data.chunks(channels) {
                            buffer.push(chunk[0] * gain); // Take first channel and apply gain
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut buffer = audio_buffer_clone.lock().unwrap();
                    if channels == 1 {
                        buffer.extend(data.iter().map(|&s| (s as f32 / 32768.0) * gain));
                    } else {
                        for chunk in data.chunks(channels) {
                            buffer.push((chunk[0] as f32 / 32768.0) * gain);
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mut buffer = audio_buffer_clone.lock().unwrap();
                    if channels == 1 {
                        buffer.extend(data.iter().map(|&s| ((s as f32 - 32768.0) / 32768.0) * gain));
                    } else {
                        for chunk in data.chunks(channels) {
                            buffer.push(((chunk[0] as f32 - 32768.0) / 32768.0) * gain);
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;

    // Process audio chunks and send them for inference
    loop {
        std::thread::sleep(std::time::Duration::from_millis(10));

        let chunk = {
            let mut buffer = audio_buffer.lock().unwrap();
            if buffer.len() >= input_chunk_size {
                let chunk: Vec<f32> = buffer.drain(..input_chunk_size).collect();
                chunk
            } else {
                continue;
            }
        };

        // Resample if needed
        let chunk = if needs_resampling {
            kaudio::resample(&chunk, sample_rate as usize, 24000)?
        } else {
            chunk
        };

        // Send chunk for processing
        if sender.send(chunk).is_err() {
            break; // Receiver dropped, stop streaming
        }
    }

    Ok(())
}

fn capture_from_microphone(duration: Option<f32>, gain: f32) -> Result<Vec<f32>> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;

    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    println!("Using input device: {}", device.name()?);

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    println!("Recording at {} Hz with {} channel(s)", sample_rate, channels);
    if gain != 1.0 {
        println!("Audio gain: {:.1}x", gain);
    }

    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let audio_buffer_clone = audio_buffer.clone();

    let err_fn = |err| eprintln!("Error in audio stream: {}", err);

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buffer = audio_buffer_clone.lock().unwrap();
                    if channels == 1 {
                        buffer.extend(data.iter().map(|&s| s * gain));
                    } else {
                        for chunk in data.chunks(channels) {
                            buffer.push(chunk[0] * gain);
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut buffer = audio_buffer_clone.lock().unwrap();
                    if channels == 1 {
                        buffer.extend(data.iter().map(|&s| (s as f32 / 32768.0) * gain));
                    } else {
                        for chunk in data.chunks(channels) {
                            buffer.push((chunk[0] as f32 / 32768.0) * gain);
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mut buffer = audio_buffer_clone.lock().unwrap();
                    if channels == 1 {
                        buffer.extend(data.iter().map(|&s| ((s as f32 - 32768.0) / 32768.0) * gain));
                    } else {
                        for chunk in data.chunks(channels) {
                            buffer.push(((chunk[0] as f32 - 32768.0) / 32768.0) * gain);
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;

    if let Some(dur) = duration {
        println!("Recording for {} seconds...", dur);
        std::thread::sleep(std::time::Duration::from_secs_f32(dur));
    } else {
        println!("Recording... Press Ctrl+C to stop.");
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    drop(stream);

    let pcm = audio_buffer.lock().unwrap().clone();

    if sample_rate != 24000 {
        println!("Resampling from {} Hz to 24000 Hz...", sample_rate);
        Ok(kaudio::resample(&pcm, sample_rate as usize, 24000)?)
    } else {
        Ok(pcm)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_devices {
        return list_audio_devices();
    }

    let device = device(args.cpu)?;
    println!("Using device: {:?}", device);

    // Load the model FIRST before capturing any audio
    println!("Loading model from repository: {}", args.hf_repo);
    println!("(This may take a few minutes on first run while downloading model files...)");
    let mut model = Model::load_from_hf(&args, &device)?;
    println!("Model loaded successfully!");
    println!();

    let use_mic = args.mic || args.in_file == "--mic";

    if use_mic {
        // Validate gain parameter
        if args.gain <= 0.0 {
            return Err(anyhow::anyhow!("Gain must be positive (got {})", args.gain));
        }
        if args.gain > 10.0 {
            println!("Warning: Gain of {:.1}x is very high and may cause distortion", args.gain);
        }

        // Use streaming mode for microphone input
        if args.duration.is_some() {
            // For fixed duration, use batch mode
            let pcm = capture_from_microphone(args.duration, args.gain)?;
            println!("Running inference");
            if let Some(ref path) = args.output {
                println!("Transcript will be appended to: {}", path);
            }
            model.run(pcm, args.output)?;
        } else {
            // For continuous recording, use streaming mode
            let (sender, receiver) = mpsc::channel();
            let gain = args.gain;

            // Spawn microphone capture thread
            std::thread::spawn(move || {
                if let Err(e) = stream_from_microphone(sender, gain) {
                    eprintln!("Microphone error: {}", e);
                }
            });

            // Run streaming inference on main thread
            model.run_streaming(receiver, args.output)?;
        }
    } else {
        // File input - use batch mode
        println!("Loading audio file from: {}", args.in_file);
        let (pcm, sample_rate) = kaudio::pcm_decode(&args.in_file)?;
        let pcm = if sample_rate != 24_000 {
            kaudio::resample(&pcm, sample_rate as usize, 24_000)?
        } else {
            pcm
        };
        println!("Running inference");
        if let Some(ref path) = args.output {
            println!("Transcript will be appended to: {}", path);
        }
        model.run(pcm, args.output)?;
    }

    Ok(())
}
