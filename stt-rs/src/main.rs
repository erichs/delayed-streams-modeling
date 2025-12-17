// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use candle::{Device, Tensor};
use clap::Parser;
use std::fs::{File, OpenOptions};
use std::io::Write as IoWrite;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
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

    /// Select audio input device by index (from --list-devices output).
    #[arg(long)]
    device: Option<usize>,

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

    /// Enable verbose debug output (buffer status, timing, chunk info).
    #[arg(long)]
    debug: bool,
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

    fn run_streaming(&mut self, audio_receiver: Receiver<Vec<f32>>, pending_chunks: Arc<AtomicU64>, output_file: Option<String>, debug: bool) -> Result<()> {
        println!("Starting real-time transcription from microphone...");
        println!("Speak now - transcription will appear as you talk.");
        println!("Press Ctrl+C to stop.");
        if let Some(ref path) = output_file {
            println!("Transcript will be appended to: {}", path);
            println!("(File will be synced every 200 bytes or 2 seconds for file watchers)");
        }
        println!();
        if debug {
            eprintln!("[DEBUG] Streaming mode initialized");
        }

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

        let mut chunk_count = 0;
        let mut total_samples = 0;
        let mut last_status_time = Instant::now();
        let status_interval = Duration::from_secs(5);

        // Transcript loss detection
        let mut last_transcript_time = Instant::now();
        let mut chunks_since_transcript = 0u64;
        let mut loss_alert_printed = false;
        let transcript_timeout = Duration::from_secs(10); // Alert after 10s without transcript
        let recv_timeout = Duration::from_secs(2); // Detect stalled audio after 2s

        loop {
            // Receive audio chunk from microphone (with timeout to detect stalls)
            let recv_start = Instant::now();
            let chunk = match audio_receiver.recv_timeout(recv_timeout) {
                Ok(chunk) => {
                    // Decrement pending counter - we received this chunk
                    pending_chunks.fetch_sub(1, Ordering::Relaxed);
                    let recv_duration = recv_start.elapsed();
                    if debug && recv_duration > Duration::from_millis(100) {
                        eprintln!("[DEBUG] Long wait for audio chunk: {:?}", recv_duration);
                    }
                    chunk
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    eprintln!("[WARNING] No audio received for {:?} - audio stream may be stalled!", recv_timeout);
                    // Print LOS indicator for timeout
                    let msg = " [LOS:audio_timeout]";
                    print!("{}", msg);
                    std::io::stdout().flush()?;
                    if let Some(ref mut writer) = file_writer {
                        writer.write(msg)?;
                    }
                    continue; // Keep trying
                },
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if debug {
                        eprintln!("[DEBUG] Audio channel closed, stopping transcription");
                    }
                    break;
                }
            };

            chunk_count += 1;
            total_samples += chunk.len();
            chunks_since_transcript += 1;
            let chunk_duration_ms = (chunk.len() as f32 / 24000.0) * 1000.0;

            if debug {
                eprintln!("[DEBUG] Chunk #{}: {} samples ({:.1}ms of audio)",
                         chunk_count, chunk.len(), chunk_duration_ms);
            }

            // Process the chunk through the model
            let pcm = Tensor::new(&chunk[..], &self.dev)?.reshape((1, 1, ()))?;

            let inference_start = Instant::now();
            let asr_msgs = self.state.step_pcm(pcm, None, &().into(), |_, _, _| ())?;
            let inference_duration = inference_start.elapsed();

            if debug {
                eprintln!("[DEBUG] Inference took: {:?}, produced {} messages",
                         inference_duration, asr_msgs.len());
            }

            if inference_duration > Duration::from_millis(200) {
                eprintln!("[WARNING] Slow inference detected: {:?}", inference_duration);
            }

            // Check for transcript loss (many chunks processed without any transcript output)
            let time_since_transcript = last_transcript_time.elapsed();
            if time_since_transcript > transcript_timeout && !loss_alert_printed {
                eprintln!("[WARNING] TRANSCRIPT LOSS DETECTED: {} chunks ({:.1}s) without transcript output!",
                         chunks_since_transcript, time_since_transcript.as_secs_f32());
                let msg = format!(" [LOS:no_transcript_{}s]", time_since_transcript.as_secs());
                print!("{}", msg);
                std::io::stdout().flush()?;
                if let Some(ref mut writer) = file_writer {
                    writer.write(&msg)?;
                }
                loss_alert_printed = true;
            }

            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        if debug {
                            eprintln!("[DEBUG] Step message - VAD scores: {:?}", &prs[2][0..3.min(prs[2].len())]);
                        }
                        if self.vad && prs[2][0] > 0.5 && !last_print_was_vad {
                            let msg = " [end of turn detected]";
                            print!("{}", msg);
                            std::io::stdout().flush()?;
                            if let Some(ref mut writer) = file_writer {
                                writer.write(msg)?;
                            }
                            last_print_was_vad = true;
                            if debug {
                                eprintln!("[DEBUG] End of turn detected");
                            }
                        }
                    }
                    moshi::asr::AsrMsg::Word { tokens, .. } => {
                        if debug {
                            eprintln!("[DEBUG] Word message - {} tokens", tokens.len());
                        }
                        let word = self
                            .text_tokenizer
                            .decode_piece_ids(tokens)
                            .unwrap_or_else(|_| String::new());

                        // Reset transcript loss tracking - we got output!
                        if loss_alert_printed {
                            eprintln!("[INFO] Transcript resumed after {:.1}s gap", time_since_transcript.as_secs_f32());
                            let msg = " [RESUMED]";
                            print!("{}", msg);
                            std::io::stdout().flush()?;
                            if let Some(ref mut writer) = file_writer {
                                writer.write(msg)?;
                            }
                        }
                        last_transcript_time = Instant::now();
                        chunks_since_transcript = 0;
                        loss_alert_printed = false;

                        // Add space before word for proper spacing
                        let output = format!(" {}", word);
                        print!("{}", output);
                        std::io::stdout().flush()?;
                        if let Some(ref mut writer) = file_writer {
                            writer.write(&output)?;
                        }
                        last_print_was_vad = false;
                        if debug {
                            eprintln!("[DEBUG] Transcribed word: '{}'", word);
                        }
                    }
                    _ => {
                        if debug {
                            eprintln!("[DEBUG] Other ASR message type");
                        }
                    }
                }
            }

            // Periodic status update (only in debug mode)
            if debug && last_status_time.elapsed() >= status_interval {
                let total_audio_seconds = total_samples as f32 / 24000.0;
                eprintln!("[STATUS] Processed {} chunks, {:.1}s of audio total, {} chunks since last transcript",
                         chunk_count, total_audio_seconds, chunks_since_transcript);
                last_status_time = Instant::now();
            }
        }

        println!();
        if let Some(ref mut writer) = file_writer {
            writer.write("\n")?;
            writer.flush()?;
        }
        if debug {
            eprintln!("[DEBUG] Transcription stopped. Total chunks processed: {}", chunk_count);
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

fn stream_from_microphone(sender: SyncSender<Vec<f32>>, pending_chunks: Arc<AtomicU64>, gain: f32, device_id: Option<usize>, debug: bool) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;

    let host = cpal::default_host();
    let device = if let Some(idx) = device_id {
        host.input_devices()?
            .nth(idx)
            .ok_or_else(|| anyhow::anyhow!("No input device at index {}", idx))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device available"))?
    };

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

    if debug {
        eprintln!("[DEBUG] Microphone streaming config:");
        eprintln!("[DEBUG]   Target chunk size: {} samples (80ms at 24kHz)", chunk_size);
        eprintln!("[DEBUG]   Input chunk size: {} samples", input_chunk_size);
        eprintln!("[DEBUG]   Needs resampling: {}", needs_resampling);
    }

    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let audio_buffer_clone = audio_buffer.clone();

    let err_fn = |err| eprintln!("[ERROR] Error in audio stream: {}", err);

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
    if debug {
        eprintln!("[DEBUG] Audio stream started");
    }

    // Process audio chunks and send them for inference
    let mut send_count = 0u64;
    let mut last_buffer_check = Instant::now();
    let mut max_buffer_size = 0;
    let mut prev_buffer_size = 0;

    loop {
        std::thread::sleep(std::time::Duration::from_millis(10));

        // CRITICAL: Keep mutex lock duration minimal to avoid blocking audio callbacks
        // All logging happens AFTER releasing the lock
        let (chunk, buffer_size, should_log) = {
            let mut buffer = audio_buffer.lock().unwrap();
            let current_size = buffer.len();

            if current_size > max_buffer_size {
                max_buffer_size = current_size;
            }

            let should_log = last_buffer_check.elapsed() >= Duration::from_secs(5);

            let chunk = if buffer.len() >= input_chunk_size {
                Some(buffer.drain(..input_chunk_size).collect::<Vec<f32>>())
            } else {
                None
            };

            (chunk, current_size, should_log)
            // Lock is released here
        };

        // Log buffer status OUTSIDE the mutex lock to prevent blocking audio callbacks
        if should_log {
            let buffer_duration_ms = (buffer_size as f32 / 24000.0) * 1000.0;

            if debug {
                let max_duration_ms = (max_buffer_size as f32 / 24000.0) * 1000.0;
                eprintln!("[DEBUG] Audio buffer status: current={} samples ({:.1}ms), max={} samples ({:.1}ms)",
                         buffer_size, buffer_duration_ms, max_buffer_size, max_duration_ms);
            }

            // Detect buffer growth trend (always warn, not debug)
            if buffer_size > prev_buffer_size + input_chunk_size {
                eprintln!("[WARNING] Buffer is GROWING! Was {} samples, now {} samples (+{} samples)",
                         prev_buffer_size, buffer_size, buffer_size - prev_buffer_size);
                eprintln!("[WARNING] This means audio is arriving faster than inference can process it!");
                eprintln!("[WARNING] Transcription will lag behind real-time by {:.1}ms and growing", buffer_duration_ms);
            }

            // Alert if buffer is getting very large (always warn)
            if buffer_duration_ms > 1000.0 {
                eprintln!("[WARNING] Buffer contains over 1 second of audio! Processing is falling behind.");
            }

            prev_buffer_size = buffer_size;
            last_buffer_check = Instant::now();
        }

        let chunk = match chunk {
            Some(c) => c,
            None => continue,
        };

        send_count += 1;
        let current_pending = pending_chunks.load(Ordering::Relaxed);
        if debug {
            eprintln!("[DEBUG] Sending chunk #{} to inference (buffer had {} samples, {} chunks pending in channel)",
                     send_count, buffer_size, current_pending);
        }

        // Resample if needed
        let chunk = if needs_resampling {
            let resample_start = Instant::now();
            let resampled = kaudio::resample(&chunk, sample_rate as usize, 24000)?;
            let resample_duration = resample_start.elapsed();
            if debug && resample_duration > Duration::from_millis(10) {
                eprintln!("[DEBUG] Resampling took: {:?}", resample_duration);
            }
            resampled
        } else {
            chunk
        };

        // Load shedding: if too many chunks pending (>25 = 2 seconds of audio), drop this chunk
        const MAX_PENDING_CHUNKS: u64 = 25;
        if current_pending > MAX_PENDING_CHUNKS {
            eprintln!("[WARNING] LOAD SHEDDING: Dropping chunk #{} - {} chunks pending (>{} threshold)",
                     send_count, current_pending, MAX_PENDING_CHUNKS);
            eprintln!("[WARNING] Inference is falling behind by ~{:.1}s", current_pending as f32 * 0.08);
            continue; // Skip this chunk to let inference catch up
        }

        // Send chunk for processing using try_send to detect backpressure
        match sender.try_send(chunk) {
            Ok(()) => {
                pending_chunks.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Full(chunk)) => {
                // Channel is full - block and send (this provides backpressure)
                eprintln!("[WARNING] Channel full, blocking send for chunk #{}", send_count);
                if sender.send(chunk).is_err() {
                    if debug {
                        eprintln!("[DEBUG] Receiver dropped, stopping microphone stream");
                    }
                    break;
                }
                pending_chunks.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                if debug {
                    eprintln!("[DEBUG] Receiver dropped, stopping microphone stream");
                }
                break;
            }
        }
    }

    if debug {
        eprintln!("[DEBUG] Microphone streaming stopped. Total chunks sent: {}", send_count);
    }
    Ok(())
}

fn capture_from_microphone(duration: Option<f32>, gain: f32, device_id: Option<usize>) -> Result<Vec<f32>> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;

    let host = cpal::default_host();
    let device = if let Some(idx) = device_id {
        host.input_devices()?
            .nth(idx)
            .ok_or_else(|| anyhow::anyhow!("No input device at index {}", idx))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device available"))?
    };

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
            let pcm = capture_from_microphone(args.duration, args.gain, args.device)?;
            println!("Running inference");
            if let Some(ref path) = args.output {
                println!("Transcript will be appended to: {}", path);
            }
            model.run(pcm, args.output)?;
        } else {
            // For continuous recording, use streaming mode
            // Use bounded channel (capacity 50 = ~4 seconds of audio) for backpressure
            const CHANNEL_CAPACITY: usize = 50;
            let (sender, receiver) = mpsc::sync_channel(CHANNEL_CAPACITY);
            let gain = args.gain;
            let debug = args.debug;
            let device_id = args.device;

            // Shared counter for pending chunks (for monitoring and load shedding)
            let pending_chunks = Arc::new(AtomicU64::new(0));
            let pending_chunks_sender = pending_chunks.clone();

            // Spawn microphone capture thread
            std::thread::spawn(move || {
                if let Err(e) = stream_from_microphone(sender, pending_chunks_sender, gain, device_id, debug) {
                    eprintln!("Microphone error: {}", e);
                }
            });

            // Run streaming inference on main thread
            model.run_streaming(receiver, pending_chunks, args.output, debug)?;
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
