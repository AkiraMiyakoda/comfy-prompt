// Copyright © 2026 Akira Miyakoda
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::io::copy;
use std::io::sink;
use std::io::stdin;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::bail;
use clap::Parser;
use eframe::egui;
use itertools::Itertools;
use serde::Deserialize;
use serde_json::value::RawValue;

const SIGNATURE_PNG: [u8; 8] = *b"\x89PNG\x0d\x0a\x1a\x0a";
const CHUNK_TYPE_TEXT: [u8; 4] = *b"tEXt";
const CHUNK_TYPE_IEND: [u8; 4] = *b"IEND";
const KEYWORD_PROMPT: [u8; 7] = *b"prompt\x00";

const MAX_TEXT_CHUNK_SIZE: u32 = 16 * 1024 * 1024;

const TEXT_NOT_CONNECTED: &str = "[Not connected]";

#[derive(Parser)]
#[command(version, about = "Extract prompts from ComfyUI PNG files")]
struct Args {
    /// Input PNG file (default: stdin)
    file: Option<PathBuf>,

    /// Display prompts in a GUI window
    #[arg(long, short)]
    gui: bool,
}

#[derive(Debug)]
struct Prompts {
    id: String,
    class_type: String,
    positive: Option<String>,
    negative: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ComfyNode<'a> {
    class_type: &'a str,

    // Keep node inputs as raw JSON to avoid parsing unused fields.
    #[serde(borrow)]
    inputs: &'a RawValue,
}

#[derive(Debug, Deserialize)]
struct TextInputs<'a> {
    // Borrow plain strings; allocate only when JSON escapes require decoding.
    #[serde(borrow)]
    text: Cow<'a, str>,
}

#[derive(Debug, Deserialize)]
struct PromptInputs<'a> {
    #[serde(borrow)]
    positive: Option<(Cow<'a, str>, i32)>,
    #[serde(borrow)]
    negative: Option<(Cow<'a, str>, i32)>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let prompts = match args.file {
        Some(path) => read_prompts(&mut File::open(path)?),
        None => read_prompts(&mut stdin()),
    }?;

    if args.gui {
        show_gui(prompts)?;
    } else {
        show_cli(prompts);
    }

    Ok(())
}

fn read_prompts(file: &mut dyn Read) -> anyhow::Result<Vec<Prompts>> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    if buf != SIGNATURE_PNG {
        bail!("Broken PNG signature");
    }

    let mut prompts: Vec<Prompts> = vec![];

    loop {
        let buf = {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            buf
        };

        let (length, chunk_type) = buf.split_at(4);
        let length = u32::from_be_bytes(length.try_into()?);

        if chunk_type == CHUNK_TYPE_IEND {
            break;
        }

        if chunk_type == CHUNK_TYPE_TEXT {
            if let Some(json_text) = read_prompt_text(file, length)? {
                append_prompts(&mut prompts, &json_text)?;
            }
        } else {
            copy(&mut (&mut *file).take((length + 4).into()), &mut sink())?;
        }
    }

    // Sort by prompt availability (positive, negative, neither),
    // then by node ID in ascending order.
    prompts.sort_by_key(|p| {
        let priority = match (p.positive.is_some(), p.negative.is_some()) {
            (true, _) => 0,
            (false, true) => 1,
            _ => 2,
        };
        (priority, p.id.parse::<i32>().unwrap_or(i32::MAX))
    });

    Ok(prompts)
}

// Read only the ComfyUI prompt metadata from PNG tEXt chunks,
// validating its CRC before returning the raw JSON bytes.
fn read_prompt_text(file: &mut dyn Read, length: u32) -> anyhow::Result<Option<Vec<u8>>> {
    if usize::try_from(length)? < KEYWORD_PROMPT.len() {
        bail!("tEXt chunk too small");
    }
    if length > MAX_TEXT_CHUNK_SIZE {
        bail!("tEXt chunk too large");
    }

    let keyword = {
        let mut keyword = [0u8; KEYWORD_PROMPT.len()];
        file.read_exact(&mut keyword)?;
        keyword
    };

    if keyword != KEYWORD_PROMPT {
        copy(
            &mut (&mut *file).take(u64::from(length + 4) - u64::try_from(keyword.len())?),
            &mut sink(),
        )?;

        return Ok(None);
    }

    let mut text = vec![0u8; usize::try_from(length)? - KEYWORD_PROMPT.len()];
    file.read_exact(&mut text)?;

    let checksum = {
        let mut checksum = [0u8; 4];
        file.read_exact(&mut checksum)?;
        u32::from_be_bytes(checksum)
    };

    // PNG CRC covers the chunk type and data, but not the length.
    let crc32 = {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&CHUNK_TYPE_TEXT);
        hasher.update(&keyword);
        hasher.update(&text);
        hasher.finalize()
    };
    if crc32 != checksum {
        bail!("CRC doesn't match")
    }

    Ok(Some(text))
}

// Resolve sampler inputs through CLIPTextEncode nodes.
// Borrow the workflow JSON during parsing and retain only
// the extracted prompt strings in the output.
fn append_prompts(prompts: &mut Vec<Prompts>, json_text: &[u8]) -> anyhow::Result<()> {
    let workflow: BTreeMap<&str, ComfyNode> = serde_json::from_slice(json_text)?;

    let texts: BTreeMap<&str, String> = workflow
        .iter()
        .filter(|(_, node)| node.class_type == "CLIPTextEncode")
        .map(|(id, node)| {
            let inputs: TextInputs = serde_json::from_str(node.inputs.get())
                .with_context(|| format!("Invalid CLIPTextEncode node: {id}"))?;
            Ok::<_, anyhow::Error>((*id, inputs.text.split_whitespace().join(" ")))
        })
        .collect::<Result<_, _>>()?;

    workflow
        .iter()
        .filter(|(_, node)| node.class_type == "KSampler" || node.class_type == "KSamplerAdvanced")
        .try_for_each(|(id, node)| -> anyhow::Result<()> {
            let inputs: PromptInputs = serde_json::from_str(node.inputs.get())
                .with_context(|| format!("Invalid KSampler/KSamplerAdvanced node: {id}"))?;

            prompts.push(Prompts {
                id: id.to_string(),
                class_type: node.class_type.to_string(),
                positive: inputs
                    .positive
                    .and_then(|p| texts.get(p.0.as_ref()).map(ToString::to_string)),
                negative: inputs
                    .negative
                    .and_then(|p| texts.get(p.0.as_ref()).map(ToString::to_string)),
            });

            Ok(())
        })?;

    Ok(())
}

fn show_cli(prompts: Vec<Prompts>) {
    for inputs in prompts {
        println!("{} [{}]", inputs.class_type, inputs.id);
        println!("Positive: {}", inputs.positive.as_deref().unwrap_or(TEXT_NOT_CONNECTED));
        println!("Nagative: {}", inputs.negative.as_deref().unwrap_or(TEXT_NOT_CONNECTED));
    }
}

fn show_gui(prompts: Vec<Prompts>) -> anyhow::Result<()> {
    let mut selected = 0;

    let theme = match dark_light::detect() {
        Ok(dark_light::Mode::Dark) => egui::Theme::Dark,
        _ => egui::Theme::Light,
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([500.0, 350.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_ui_native("comfy-prompt", options, move |ui, _frame| {
        ui.ctx().set_theme(theme);

        egui::CentralPanel::default().show(ui, |ui| {
            if prompts.is_empty() {
                ui.label("No prompts found");
                return;
            }

            egui::ComboBox::from_label("Node")
                .selected_text(format!("{} [{}]", prompts[selected].class_type, prompts[selected].id))
                .show_ui(ui, |ui| {
                    for (i, prompt) in prompts.iter().enumerate() {
                        ui.selectable_value(&mut selected, i, format!("{} [{}]", prompt.class_type, prompt.id));
                    }
                });

            ui.add_space(12.0);

            let prompt = &prompts[selected];

            let frame = egui::Frame::new()
                .fill(ui.visuals().extreme_bg_color)
                .stroke(egui::Stroke::new(
                    1.0,
                    ui.visuals().widgets.noninteractive.bg_stroke.color,
                ))
                .corner_radius(4)
                .inner_margin(8);

            let spacing = ui.spacing().item_spacing.y;
            let header_height = ui.spacing().interact_size.y;
            let height = (ui.available_height() - 2.0 * header_height - 3.0 * spacing - 8.0) / 2.0;

            ui.horizontal(|ui| {
                ui.label("Positive Prompt");

                if ui.button("Copy").clicked()
                    && let Some(text) = &prompt.positive
                {
                    ui.ctx().copy_text(text.clone());
                }
            });

            let mut positive = prompt.positive.as_deref().unwrap_or(TEXT_NOT_CONNECTED);
            ui.add_sized(
                [ui.available_width(), height],
                egui::TextEdit::multiline(&mut positive).frame(frame),
            );

            ui.add_space(8.0);

            ui.horizontal(|ui| {
                ui.label("Negative Prompt");

                if ui.button("Copy").clicked()
                    && let Some(text) = &prompt.negative
                {
                    ui.ctx().copy_text(text.clone());
                }
            });

            let mut negative = prompt.negative.as_deref().unwrap_or(TEXT_NOT_CONNECTED);
            ui.add_sized(
                [ui.available_width(), height],
                egui::TextEdit::multiline(&mut negative).frame(frame),
            );
        });
    })?;

    Ok(())
}
