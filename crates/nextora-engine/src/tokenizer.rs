//! Byte-level BPE assembled directly from this engine's GGUF metadata.
//!
//! Hugging Face's `tokenizers` crate performs BPE; this module owns metadata
//! validation, special-token registration and generation stop IDs. It does not
//! call RustingStrata. Qwen2's published pre-tokenization rule is also used by
//! Qwen3: https://huggingface.co/Qwen/Qwen3-0.6B/tree/main .
//!
//! Embedded chat-template evaluation belongs to `protocol.rs`.
//! `render_qwen_chatml` is the explicit text-only helper provided here;
//! raw encoding and decoding never apply a chat wrapper.

use crate::gguf::{Gguf, Value};
use anyhow::{anyhow, bail, ensure, Context, Result};
use std::collections::{BTreeSet, HashMap, HashSet};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::{
    byte_level::ByteLevel,
    sequence::Sequence,
    split::{Split, SplitPattern},
};
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior};

// The spelling is part of the model's tokenizer contract: notably, each digit
// is split individually. Replacing it with the GPT-2 default changes token IDs.
const QWEN2_SPLIT: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    bpe: tokenizers::Tokenizer,
    vocabulary_size: usize,
    // Packed byte spellings plus a sentinel offset avoid one allocation per
    // vocabulary entry and any token-string lookup during streaming.
    decode_bytes: Vec<u8>,
    decode_offsets: Vec<usize>,
    end_tokens: BTreeSet<u32>,
    pre_tokenizer: String,
    has_chat_template: bool,
}

fn text_array(g: &Gguf, key: &str) -> Result<Vec<String>> {
    let values = g
        .meta
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("{key} must be a GGUF string array"))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("{key}[{index}] must be a string"))
        })
        .collect()
}

/// GPT-2's bijection from all byte values to visible Unicode symbols. Byte
/// values absent from the three visible ranges are assigned consecutive code
/// points starting at U+0100, in increasing byte order.
fn byte_symbol(value: u8) -> char {
    let visible = |b: u8| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let point = if visible(value) {
        u32::from(value)
    } else {
        256 + (0..value).filter(|b| !visible(*b)).count() as u32
    };
    char::from_u32(point).expect("byte alphabet consists of valid Unicode scalars")
}

fn decoding_table(spellings: &[String]) -> Result<(Vec<u8>, Vec<usize>)> {
    let reverse: HashMap<char, u8> = (0..=u8::MAX)
        .map(|byte| (byte_symbol(byte), byte))
        .collect();
    let capacity = spellings.iter().try_fold(0usize, |n, spelling| {
        n.checked_add(spelling.len())
            .context("decoded vocabulary byte count overflow")
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .context("allocate decoded vocabulary bytes")?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(
            spellings
                .len()
                .checked_add(1)
                .context("decoded vocabulary offset count overflow")?,
        )
        .context("allocate decoded vocabulary offsets")?;
    offsets.push(0);
    for spelling in spellings {
        let start = bytes.len();
        // Match tokenizers' ByteLevel decoder: map all characters through the
        // byte alphabet, or use the entire token's raw UTF-8 if any character
        // is absent. This also preserves its added/control-token semantics.
        for symbol in spelling.chars() {
            if let Some(&byte) = reverse.get(&symbol) {
                bytes.push(byte);
            } else {
                bytes.truncate(start);
                bytes.extend_from_slice(spelling.as_bytes());
                break;
            }
        }
        offsets.push(bytes.len());
    }
    Ok((bytes, offsets))
}

// Append every determined scalar or invalid sequence exactly as
// String::from_utf8_lossy does. Only a potentially valid incomplete suffix is
// deferred. UTF-8's maximum width makes that suffix at most three bytes.
fn append_utf8(
    mut bytes: &[u8],
    output: &mut String,
    pending: &mut [u8; 3],
    pending_len: &mut usize,
) {
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                output.push_str(valid);
                return;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    output.push_str(
                        std::str::from_utf8(&bytes[..valid]).expect("validated UTF-8 prefix"),
                    );
                }
                bytes = &bytes[valid..];
                match error.error_len() {
                    Some(invalid) => {
                        output.push('\u{fffd}');
                        bytes = &bytes[invalid..];
                    }
                    None => {
                        debug_assert!(bytes.len() <= pending.len());
                        pending[..bytes.len()].copy_from_slice(bytes);
                        *pending_len = bytes.len();
                        return;
                    }
                }
            }
        }
    }
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g
            .meta
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .context("tokenizer.ggml.model must be a string")?;
        ensure!(
            model == "gpt2",
            "Forge engine supports GPT-2 byte BPE tokenizer metadata only; found {model:?}"
        );
        let pre = g
            .meta
            .get("tokenizer.ggml.pre")
            .and_then(Value::as_str)
            .context("tokenizer.ggml.pre must explicitly identify the pre-tokenizer")?;
        ensure!(
            matches!(pre, "qwen2" | "gpt2" | "default"),
            "unsupported pre-tokenizer {pre:?}; supported: qwen2, gpt2, default"
        );

        for name in [
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token",
        ] {
            if let Some(value) = g.meta.get(name) {
                let insert = value
                    .as_bool()
                    .with_context(|| format!("{name} must be boolean"))?;
                ensure!(
                    !insert,
                    "{name}=true is unsupported by the raw-prompt tokenizer; automatic token insertion is not silently ignored"
                );
            }
        }

        let spellings = text_array(g, "tokenizer.ggml.tokens")?;
        ensure!(
            !spellings.is_empty() && spellings.len() <= u32::MAX as usize,
            "vocabulary must contain between 1 and u32::MAX tokens"
        );
        let mut vocabulary = Vocab::with_capacity(spellings.len());
        for (index, spelling) in spellings.iter().enumerate() {
            ensure!(
                !spelling.is_empty(),
                "empty vocabulary spelling at token {index}"
            );
            ensure!(
                vocabulary.insert(spelling.clone(), index as u32).is_none(),
                "duplicate vocabulary spelling at token {index}"
            );
        }
        for byte in 0..=u8::MAX {
            ensure!(
                vocabulary.contains_key(&byte_symbol(byte).to_string()),
                "byte-level vocabulary is missing base byte {byte}; encoding would lose input bytes"
            );
        }

        let types = g
            .meta
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .context("tokenizer.ggml.token_type must be an integer array")?;
        ensure!(
            types.len() == spellings.len(),
            "token_type length does not match vocabulary length"
        );
        let mut literal_tokens = Vec::new();
        let mut kinds = Vec::with_capacity(types.len());
        for (index, kind) in types.iter().enumerate() {
            let kind = kind
                .as_u64()
                .with_context(|| format!("token_type[{index}] must be a nonnegative integer"))?;
            ensure!(
                (1..=6).contains(&kind),
                "unsupported GGUF token kind {kind} at token {index}"
            );
            kinds.push(kind);
            if kind == 3 || kind == 4 {
                literal_tokens.push(
                    AddedToken::from(spellings[index].clone(), kind == 3)
                        .normalized(false)
                        .single_word(false)
                        .lstrip(false)
                        .rstrip(false),
                );
            }
        }

        let merge_lines = text_array(g, "tokenizer.ggml.merges")?;
        let mut seen_merges = HashSet::new();
        let mut merges = Vec::with_capacity(merge_lines.len());
        for (index, line) in merge_lines.iter().enumerate() {
            let (left, right) = line.split_once(' ').with_context(|| {
                format!("BPE merge {index} must contain two symbols separated by one space")
            })?;
            ensure!(
                !left.is_empty() && !right.is_empty() && !right.contains(' '),
                "malformed BPE merge {index}"
            );
            ensure!(
                vocabulary.contains_key(left)
                    && vocabulary.contains_key(right)
                    && vocabulary.contains_key(&format!("{left}{right}")),
                "BPE merge {index} refers to symbols missing from the vocabulary"
            );
            let pair = (left.to_owned(), right.to_owned());
            ensure!(
                seen_merges.insert(pair.clone()),
                "duplicate BPE merge {index}"
            );
            merges.push(pair);
        }

        let model = BPE::builder()
            .vocab_and_merges(vocabulary, merges)
            .build()
            .map_err(|error| anyhow!("cannot construct GGUF BPE tokenizer: {error}"))?;
        let mut bpe = tokenizers::Tokenizer::new(model);
        if pre == "qwen2" {
            let split = Split::new(
                SplitPattern::Regex(QWEN2_SPLIT.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|error| anyhow!("cannot construct Qwen pre-tokenizer: {error}"))?;
            bpe.with_pre_tokenizer(Some(Sequence::new(vec![
                PreTokenizerWrapper::Split(split),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
            ])));
        } else {
            bpe.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
        }
        bpe.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        // register_literal_tokens keeps existing GGUF IDs; it does not append a
        // separate vocabulary. USER_DEFINED tokens stay non-special literals.
        bpe.add_tokens(literal_tokens)
            .map_err(|error| anyhow!("cannot register GGUF literal tokens: {error}"))?;
        ensure!(
            bpe.get_vocab_size(true) == spellings.len(),
            "literal token registration changed the vocabulary size"
        );
        for (index, kind) in kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| matches!(**kind, 3 | 4))
        {
            ensure!(
                bpe.token_to_id(&spellings[index]) == Some(index as u32),
                "literal token registration changed GGUF token ID {index} (kind {kind})"
            );
        }

        let mut end_tokens = BTreeSet::new();
        let mut add_end = |value: &Value, source: &str| -> Result<()> {
            let id = value
                .as_u64()
                .with_context(|| format!("{source} must contain integer token IDs"))?;
            ensure!(
                id < spellings.len() as u64,
                "{source} token {id} is outside the vocabulary"
            );
            ensure!(
                matches!(kinds[id as usize], 3 | 4),
                "{source} token {id} is not a control or user-defined token"
            );
            end_tokens.insert(id as u32);
            Ok(())
        };
        if let Some(value) = g.meta.get("tokenizer.ggml.eos_token_id") {
            add_end(value, "tokenizer.ggml.eos_token_id")?;
        }
        if let Some(value) = g.meta.get("tokenizer.ggml.eog_token_ids") {
            for value in value
                .as_array()
                .context("tokenizer.ggml.eog_token_ids must be an array")?
            {
                add_end(value, "tokenizer.ggml.eog_token_ids")?;
            }
        }
        for name in ["<|endoftext|>", "<|im_end|>"] {
            if let Some(id) = bpe.token_to_id(name) {
                ensure!(
                    matches!(kinds[id as usize], 3 | 4),
                    "generation stop marker {name} must be a literal/control token"
                );
                end_tokens.insert(id);
            }
        }
        let has_chat_template = match g.meta.get("tokenizer.chat_template") {
            None => false,
            Some(Value::String(template)) => !template.is_empty(),
            Some(_) => bail!("tokenizer.chat_template must be a string when present"),
        };
        let (decode_bytes, decode_offsets) = decoding_table(&spellings)?;
        Ok(Self {
            bpe,
            vocabulary_size: spellings.len(),
            decode_bytes,
            decode_offsets,
            end_tokens,
            pre_tokenizer: pre.to_owned(),
            has_chat_template,
        })
    }

    /// Encode exactly the supplied string. No BOS/EOS or chat wrapper is added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .bpe
            .encode(text, false)
            .map_err(|error| anyhow!("tokenization failed: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode with control spellings retained. Invalid IDs are rejected rather
    /// than silently omitted. Incomplete UTF-8 byte sequences use U+FFFD, as in
    /// the byte-level decoder; streaming callers must retain undecoded bytes.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        ensure!(
            ids.iter().all(|id| (*id as usize) < self.vocabulary_size),
            "decode token ID is outside the vocabulary"
        );
        self.bpe
            .decode(ids, false)
            .map_err(|error| anyhow!("token decoding failed: {error}"))
    }

    /// Incrementally decode one sequence, retaining control-token spellings.
    /// Append each returned chunk once; `None` means the decoder is waiting for
    /// complete UTF-8. Per-token bytes are prepared at tokenizer load; each
    /// stream retains at most three pending bytes, never token history.
    ///
    /// At generation end, call `decode` once on the generated IDs and reconcile
    /// the remaining suffix: a trailing incomplete byte sequence may still be
    /// buffered. An invalid ID is rejected
    /// before it can alter this stream's state. Create a new closure per answer.
    pub fn stream_decoder(&self) -> impl FnMut(u32) -> Result<Option<String>> + '_ {
        let mut pending = [0u8; 3];
        let mut pending_len = 0usize;
        move |id| {
            let index = id as usize;
            ensure!(
                index < self.vocabulary_size,
                "decode token ID is outside the vocabulary"
            );
            let mut bytes =
                &self.decode_bytes[self.decode_offsets[index]..self.decode_offsets[index + 1]];
            if pending_len == 0 {
                if let Ok(valid) = std::str::from_utf8(bytes) {
                    return Ok((!valid.is_empty()).then(|| valid.to_owned()));
                }
            }
            let mut output = String::new();
            while pending_len > 0 && !bytes.is_empty() {
                let mut joined = [0u8; 4];
                joined[..pending_len].copy_from_slice(&pending[..pending_len]);
                let copied = bytes.len().min(joined.len() - pending_len);
                joined[pending_len..pending_len + copied].copy_from_slice(&bytes[..copied]);
                let joined_len = pending_len + copied;
                bytes = &bytes[copied..];
                pending_len = 0;
                append_utf8(
                    &joined[..joined_len],
                    &mut output,
                    &mut pending,
                    &mut pending_len,
                );
            }
            if !bytes.is_empty() {
                append_utf8(bytes, &mut output, &mut pending, &mut pending_len);
            }
            Ok((!output.is_empty()).then_some(output))
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocabulary_size
    }
    pub fn token_id(&self, spelling: &str) -> Option<u32> {
        self.bpe.token_to_id(spelling)
    }
    pub fn is_eos(&self, id: u32) -> bool {
        self.end_tokens.contains(&id)
    }
    pub fn eos_ids(&self) -> Vec<u32> {
        self.end_tokens.iter().copied().collect()
    }
    pub fn pre_tokenizer(&self) -> &str {
        &self.pre_tokenizer
    }
    pub fn has_embedded_chat_template(&self) -> bool {
        self.has_chat_template
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Explicit text-only Qwen3 ChatML rendering. This is not a Jinja interpreter,
/// does not claim to support arbitrary GGUF templates, and is never selected by
/// `encode`. Tool calls, multimodal content and assistant continuation are not
/// supported by this helper. Leading developer messages join the system text.
pub fn render_qwen_chatml(messages: &[ChatMessage], enable_thinking: bool) -> Result<String> {
    ensure!(!messages.is_empty(), "ChatML requires at least one message");
    ensure!(
        messages
            .last()
            .is_some_and(|message| message.role == "user"),
        "text-only ChatML generation requires a final user message; assistant continuation and tool turns are unsupported"
    );
    let mut rendered = String::new();
    let mut instructions = Vec::new();
    let mut first_dialogue = 0;
    while first_dialogue < messages.len()
        && matches!(
            messages[first_dialogue].role.as_str(),
            "system" | "developer"
        )
    {
        instructions.push(messages[first_dialogue].content.as_str());
        first_dialogue += 1;
    }
    if !instructions.is_empty() {
        rendered.push_str("<|im_start|>system\n");
        rendered.push_str(&instructions.join("\n\n"));
        rendered.push_str("<|im_end|>\n");
    }
    for message in &messages[first_dialogue..] {
        ensure!(
            matches!(message.role.as_str(), "user" | "assistant"),
            "unsupported text-only ChatML role {:?}; system/developer messages must lead the conversation",
            message.role
        );
        rendered.push_str("<|im_start|>");
        rendered.push_str(&message.role);
        rendered.push('\n');
        // All assistant turns precede the final user query. Qwen3's standard
        // template removes their old reasoning while retaining answer text.
        let content = if message.role == "assistant" {
            message
                .content
                .rsplit_once("</think>")
                .map_or(message.content.as_str(), |(_, answer)| {
                    answer.trim_start_matches('\n')
                })
        } else {
            message.content.as_str()
        };
        rendered.push_str(content);
        rendered.push_str("<|im_end|>\n");
    }
    rendered.push_str("<|im_start|>assistant\n");
    if !enable_thinking {
        rendered.push_str("<think>\n\n</think>\n\n");
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Gguf {
        Gguf::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-qwen3.gguf"),
        )
        .unwrap()
    }

    fn merged_fixture() -> (Gguf, u32) {
        let mut gguf = fixture();
        let Value::Array(tokens) = gguf.meta.get_mut("tokenizer.ggml.tokens").unwrap() else {
            panic!()
        };
        let first = tokens.len() as u32;
        tokens.extend(
            [
                "he",
                "hel",
                "hell",
                "hello",
                "12",
                "<custom>",
                "<literal:λ>",
            ]
            .into_iter()
            .map(|text| Value::String(text.into())),
        );
        let Value::Array(kinds) = gguf.meta.get_mut("tokenizer.ggml.token_type").unwrap() else {
            panic!()
        };
        kinds.extend([1, 1, 1, 1, 1, 4, 4].into_iter().map(Value::U32));
        gguf.meta.insert(
            "tokenizer.ggml.merges".into(),
            Value::Array(
                ["h e", "he l", "hel l", "hell o", "1 2"]
                    .into_iter()
                    .map(|pair| Value::String(pair.into()))
                    .collect(),
            ),
        );
        (gguf, first)
    }

    #[test]
    fn bytes_unicode_specials_and_invalid_ids() {
        let tokenizer = Tokenizer::from_gguf(&fixture()).unwrap();
        for text in [
            "hello world\n",
            "£ café 日本語 😀",
            "\0\t\r\n",
            "<|im_start|>user\nhey<|im_end|>",
            "<think>thought</think>",
        ] {
            assert_eq!(
                tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(),
                text
            );
        }
        assert_eq!(tokenizer.encode("<|im_end|>").unwrap(), vec![258]);
        assert!(tokenizer.is_eos(256) && tokenizer.is_eos(258));
        assert!(!tokenizer.is_eos(65));
        assert!(tokenizer.decode(&[tokenizer.vocab_size() as u32]).is_err());
        assert_eq!(tokenizer.encode("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn streaming_unicode_waits_for_complete_bytes_and_matches_final_decode() {
        let tokenizer = Tokenizer::from_gguf(&fixture()).unwrap();
        for scalar in ['é', '£', '日', '😀'] {
            let mut decode = tokenizer.stream_decoder();
            let text = scalar.to_string();
            for (index, byte) in text.bytes().enumerate() {
                let id = tokenizer.token_id(&byte_symbol(byte).to_string()).unwrap();
                let chunk = decode(id).unwrap();
                if index + 1 == text.len() {
                    assert_eq!(chunk.as_deref(), Some(text.as_str()));
                } else {
                    assert_eq!(chunk, None);
                }
            }
            assert_eq!(decode(u32::from(b'!')).unwrap().as_deref(), Some("!"));
        }
        let text = "let café = \"日本語 😀\";\n\t£\0<|im_end|>";
        let ids = tokenizer.encode(text).unwrap();
        let mut decode = tokenizer.stream_decoder();
        let mut streamed = String::new();
        for &id in &ids {
            if let Some(chunk) = decode(id).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        assert_eq!(streamed, text);
        assert_eq!(streamed, tokenizer.decode(&ids).unwrap());
    }

    #[test]
    fn streaming_preserves_merged_added_and_control_token_spellings() {
        let (gguf, first) = merged_fixture();
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let text = "hello<custom><|im_end|><literal:λ>12";
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(
            ids,
            vec![
                first + 3,
                first + 5,
                258,
                first + 6,
                u32::from(b'1'),
                u32::from(b'2')
            ]
        );
        let mut decode = tokenizer.stream_decoder();
        let chunks = ids
            .iter()
            .filter_map(|&id| decode(id).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            chunks,
            ["hello", "<custom>", "<|im_end|>", "<literal:λ>", "1", "2"]
        );
        assert_eq!(chunks.concat(), tokenizer.decode(&ids).unwrap());
    }

    #[test]
    fn streaming_rejects_invalid_ids_before_changing_pending_bytes() {
        let tokenizer = Tokenizer::from_gguf(&fixture()).unwrap();
        let lead = tokenizer.token_id(&byte_symbol(0xc3).to_string()).unwrap();
        let continuation = tokenizer.token_id(&byte_symbol(0xa9).to_string()).unwrap();
        let mut decode = tokenizer.stream_decoder();
        assert_eq!(decode(u32::from(b'A')).unwrap().as_deref(), Some("A"));
        assert_eq!(decode(lead).unwrap(), None);
        for invalid in [tokenizer.vocab_size() as u32, u32::MAX] {
            assert!(decode(invalid)
                .unwrap_err()
                .to_string()
                .contains("outside the vocabulary"));
        }
        assert_eq!(decode(continuation).unwrap().as_deref(), Some("é"));
        assert_eq!(decode(u32::from(b'B')).unwrap().as_deref(), Some("B"));
        assert_eq!(
            tokenizer
                .decode(&[u32::from(b'A'), lead, continuation, u32::from(b'B')])
                .unwrap(),
            "AéB"
        );
    }

    #[test]
    fn streaming_final_decode_flushes_only_incomplete_suffixes() {
        let tokenizer = Tokenizer::from_gguf(&fixture()).unwrap();
        let lead = tokenizer.token_id(&byte_symbol(0xc3).to_string()).unwrap();
        let mut incomplete = tokenizer.stream_decoder();
        assert_eq!(incomplete(u32::from(b'A')).unwrap().as_deref(), Some("A"));
        assert_eq!(incomplete(lead).unwrap(), None);
        assert_eq!(
            tokenizer
                .decode(&[u32::from(b'A'), lead])
                .unwrap()
                .strip_prefix('A'),
            Some("�")
        );
        let ids = tokenizer.encode("�").unwrap();
        let mut replacement = tokenizer.stream_decoder();
        for (index, &id) in ids.iter().enumerate() {
            let output = replacement(id).unwrap();
            if index + 1 == ids.len() {
                assert_eq!(output.as_deref(), Some("�"));
            } else {
                assert_eq!(output, None);
            }
        }
        assert_eq!(tokenizer.decode(&ids).unwrap(), "�");
        assert_eq!(replacement(u32::from(b' ')).unwrap().as_deref(), Some(" "));
        let invalid = tokenizer.token_id(&byte_symbol(0xff).to_string()).unwrap();
        assert_eq!(replacement(invalid).unwrap().as_deref(), Some("�"));
    }

    fn assert_stream_agrees(tokenizer: &Tokenizer, ids: &[u32]) {
        let mut decode = tokenizer.stream_decoder();
        let mut streamed = String::new();
        for &id in ids {
            if let Some(chunk) = decode(id).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        let full = tokenizer.decode(ids).unwrap();
        let suffix = full.strip_prefix(&streamed).unwrap_or_else(|| {
            panic!("stream differs for IDs {ids:?}: {streamed:?} versus {full:?}")
        });
        // Only one incomplete UTF-8 scalar may remain for the final flush.
        assert!(
            suffix.is_empty() || suffix == "�",
            "too much buffered for IDs {ids:?}: {suffix:?}"
        );
    }

    #[test]
    fn streaming_matches_reference_for_every_single_and_two_byte_pattern() {
        let tokenizer = Tokenizer::from_gguf(&fixture()).unwrap();
        let ids: [u32; 256] = std::array::from_fn(|byte| {
            tokenizer
                .token_id(&byte_symbol(byte as u8).to_string())
                .unwrap()
        });
        for &first in &ids {
            assert_stream_agrees(&tokenizer, &[first]);
            for &second in &ids {
                assert_stream_agrees(&tokenizer, &[first, second]);
            }
        }
    }

    #[test]
    fn streaming_matches_reference_for_mixed_invalid_and_unicode_sequences() {
        let tokenizer = Tokenizer::from_gguf(&fixture()).unwrap();
        let byte_ids: [u32; 256] = std::array::from_fn(|byte| {
            tokenizer
                .token_id(&byte_symbol(byte as u8).to_string())
                .unwrap()
        });
        for seed in 1..=48u64 {
            let mut state = seed;
            let mut bytes = Vec::new();
            for _ in 0..384 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match (state >> 32) & 7 {
                    0 => bytes.extend_from_slice("€".as_bytes()),
                    1 => bytes.extend_from_slice("😀".as_bytes()),
                    2 => bytes.extend_from_slice("日本語".as_bytes()),
                    3 => bytes.extend_from_slice(&[0xf4, 0x90, 0x80, 0x80]),
                    4 => bytes.extend_from_slice(&[0xe2, 0x82]),
                    5 => bytes.extend_from_slice(&[0xed, 0xa0, b'A']),
                    _ => bytes.push((state >> 40) as u8),
                }
            }
            let ids = bytes
                .iter()
                .map(|&b| byte_ids[b as usize])
                .collect::<Vec<_>>();
            assert_stream_agrees(&tokenizer, &ids);
        }
    }

    #[test]
    fn streaming_handles_partial_scalars_inside_merged_tokens_and_raw_literals() {
        let mut gguf = fixture();
        let Value::Array(tokens) = gguf.meta.get_mut("tokenizer.ggml.tokens").unwrap() else {
            panic!()
        };
        let first = tokens.len() as u32;
        for bytes in [
            &[b'A', 0xe2][..],
            &[0x82, 0xac, 0xf0][..],
            &[0x9f, 0x98, 0x80, 0xff, b'Z'][..],
        ] {
            tokens.push(Value::String(
                bytes.iter().map(|&b| byte_symbol(b)).collect(),
            ));
        }
        tokens.push(Value::String("éλ".into()));
        let Value::Array(kinds) = gguf.meta.get_mut("tokenizer.ggml.token_type").unwrap() else {
            panic!()
        };
        kinds.extend([1, 1, 1, 4].into_iter().map(Value::U32));
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let ids = [first, first + 1, first + 2, first + 3];
        assert_eq!(tokenizer.decode(&ids).unwrap(), "A€😀�Zéλ");
        let mut decode = tokenizer.stream_decoder();
        assert_eq!(decode(first).unwrap().as_deref(), Some("A"));
        assert_eq!(decode(first + 1).unwrap().as_deref(), Some("€"));
        assert_eq!(decode(first + 2).unwrap().as_deref(), Some("😀�Z"));
        assert_eq!(decode(first + 3).unwrap().as_deref(), Some("éλ"));
        assert_stream_agrees(&tokenizer, &ids);
    }

    #[test]
    fn malformed_metadata_is_rejected_before_tokenization() {
        for mutate in [
            ("tokenizer.ggml.pre", Value::String("unknown".into())),
            ("tokenizer.ggml.add_bos_token", Value::Bool(true)),
            ("tokenizer.ggml.token_type", Value::Array(vec![])),
            ("tokenizer.ggml.eos_token_id", Value::U32(u32::MAX)),
        ] {
            let mut gguf = fixture();
            gguf.meta.insert(mutate.0.into(), mutate.1);
            assert!(Tokenizer::from_gguf(&gguf).is_err(), "{}", mutate.0);
        }
        let mut gguf = fixture();
        let Value::Array(tokens) = gguf.meta.get_mut("tokenizer.ggml.tokens").unwrap() else {
            panic!()
        };
        tokens[1] = tokens[0].clone();
        assert!(Tokenizer::from_gguf(&gguf).is_err());
    }

    #[test]
    fn merge_ranks_qwen_digit_splitting_and_user_defined_literals() {
        let (mut gguf, first) = merged_fixture();
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let ids = tokenizer.encode("hello<custom>12").unwrap();
        assert_eq!(
            ids,
            vec![first + 3, first + 5, u32::from(b'1'), u32::from(b'2')]
        );
        assert_eq!(tokenizer.decode(&ids).unwrap(), "hello<custom>12");
        // The GPT-2 rule groups digits, whereas the Qwen rule separates each.
        gguf.meta
            .insert("tokenizer.ggml.pre".into(), Value::String("gpt2".into()));
        assert_eq!(
            Tokenizer::from_gguf(&gguf).unwrap().encode("12").unwrap(),
            vec![first + 4]
        );
    }

    #[test]
    fn chatml_is_explicit_text_only_and_removes_old_reasoning() {
        let messages = vec![
            ChatMessage {
                role: "developer".into(),
                content: "Be precise".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "First".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>old reasoning</think>\nAnswer".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Next".into(),
            },
        ];
        let rendered = render_qwen_chatml(&messages, false).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe precise<|im_end|>\n<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\nAnswer<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(render_qwen_chatml(&messages, true)
            .unwrap()
            .ends_with("<|im_start|>assistant\n"));
        assert!(render_qwen_chatml(&[], false).is_err());
        assert!(render_qwen_chatml(
            &[ChatMessage {
                role: "tool".into(),
                content: "value".into()
            }],
            false
        )
        .is_err());
    }
}
