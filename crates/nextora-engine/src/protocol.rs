//! Native chat preparation and output parsing. The model's embedded template is authoritative.
use crate::{
    gguf::Gguf,
    tokenizer::{ChatMessage, render_qwen_chatml},
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub struct ChatTemplate {
    env: Option<minijinja::Environment<'static>>,
}
impl ChatTemplate {
    pub fn new(g: &Gguf) -> Result<Self> {
        let Some(source) = g
            .meta
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
        else {
            return Ok(Self { env: None });
        };
        let mut env = minijinja::Environment::new();
        env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
        env.set_fuel(Some(2_000_000));
        env.set_recursion_limit(128);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Transformers overrides Jinja's HTML-escaping tojson. Match its default
        // UTF-8 spelling and comma/colon spacing for the supported model template.
        env.add_filter("tojson", template_json);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template_owned("chat", without_generation_annotations(source)?)?;
        Ok(Self { env: Some(env) })
    }
    pub fn render(&self, request: &ChatRequest) -> Result<String> {
        if let Some(env) = &self.env {
            let mut values = json!({"messages":request.messages,"add_generation_prompt":true,
                "enable_thinking":request.thinking,"preserve_thinking":true,"add_vision_id":false,
                "bos_token":"","eos_token":"<|im_end|>"});
            if !request.tools.is_empty() {
                values["tools"] = Value::Array(request.tools.clone());
            }
            let text = env
                .get_template("chat")?
                .render(minijinja::value::Serde(values))?;
            ensure!(
                text.len() <= 16 * 1024 * 1024,
                "rendered prompt exceeds 16 MiB"
            );
            Ok(text)
        } else {
            ensure!(
                request.tools.is_empty()
                    && request
                        .messages
                        .iter()
                        .all(|m| m.get("tool_calls").is_none()),
                "tool use requires an embedded model chat template"
            );
            let messages = request
                .messages
                .iter()
                .map(|m| ChatMessage {
                    role: m["role"].as_str().unwrap().into(),
                    content: m["content"].as_str().unwrap_or("").into(),
                })
                .collect::<Vec<_>>();
            render_qwen_chatml(&messages, request.thinking)
        }
    }
}

fn template_json(value: minijinja::Value) -> std::result::Result<String, minijinja::Error> {
    struct Formatter;
    impl serde_json::ser::Formatter for Formatter {
        fn begin_array_value<W: std::io::Write + ?Sized>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_key<W: std::io::Write + ?Sized>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_value<W: std::io::Write + ?Sized>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }
    let mut bytes = Vec::new();
    serde::Serialize::serialize(
        &value,
        &mut serde_json::Serializer::with_formatter(&mut bytes, Formatter),
    )
    .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))?;
    String::from_utf8(bytes)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))
}

// Hugging Face generation annotations mark training spans; they emit no text.
// Replace only these two exact tags with a no-op retaining Jinja whitespace flags.
fn without_generation_annotations(source: &str) -> Result<String> {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    let mut depth = 0usize;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let end = rest.find("%}").context("unterminated Jinja tag")? + 2;
        let tag = &rest[..end];
        let body = tag[2..tag.len() - 2].trim().trim_matches('-').trim();
        if body == "generation" || body == "endgeneration" {
            if body == "generation" {
                depth += 1;
            } else {
                ensure!(depth > 0, "unmatched endgeneration");
                depth -= 1;
            }
            out.push_str(if tag.starts_with("{%-") { "{%-" } else { "{%" });
            out.push_str(" set __forge_generation_annotation = true ");
            out.push_str(if tag.ends_with("-%}") { "-%}" } else { "%}" });
        } else {
            out.push_str(tag);
        }
        rest = &rest[end..];
    }
    ensure!(depth == 0, "unclosed generation annotation");
    out.push_str(rest);
    Ok(out)
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub max_tokens: usize,
    pub stream: bool,
    pub include_usage: bool,
    pub thinking: bool,
    pub opens_reasoning: bool,
    pub stops: Vec<String>,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub parallel_tool_calls: bool,
}

fn bool_value(v: &Value, key: &str, default: bool) -> Result<bool> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x
            .as_bool()
            .with_context(|| format!("{key} must be boolean")),
    }
}
fn float_value(v: &Value, key: &str, default: f32, min: f32, max: f32) -> Result<f32> {
    let n = match v.get(key) {
        None => default,
        Some(x) => x
            .as_f64()
            .with_context(|| format!("{key} must be a number"))? as f32,
    };
    ensure!(
        n.is_finite() && n >= min && n <= max,
        "{key} must be in {min}..={max}"
    );
    Ok(n)
}
fn name_ok(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_-.".contains(c))
}

impl ChatRequest {
    pub fn parse(v: &Value, max_output: usize) -> Result<Self> {
        let object = v.as_object().context("request must be a JSON object")?;
        let allowed = [
            "model",
            "messages",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "stop",
            "temperature",
            "top_p",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "user",
            "chat_template_kwargs",
        ];
        for k in object.keys() {
            ensure!(
                allowed.contains(&k.as_str()),
                "unsupported request parameter: {k}"
            );
        }
        if let Some(n) = v.get("n") {
            ensure!(n.as_u64() == Some(1), "only n=1 is supported");
        }
        let model = v["model"]
            .as_str()
            .filter(|s| !s.is_empty())
            .context("model must name a loaded/catalogue model")?
            .to_owned();
        let source = v["messages"]
            .as_array()
            .context("messages must be an array")?;
        ensure!(
            !source.is_empty() && source.len() <= 4096,
            "messages must contain 1..=4096 entries"
        );
        let mut messages = Vec::with_capacity(source.len());
        let mut pending = BTreeMap::new();
        for original in source {
            let mut m = original
                .as_object()
                .context("message must be an object")?
                .clone();
            let mut role = m
                .get("role")
                .and_then(Value::as_str)
                .context("message role is required")?
                .to_owned();
            ensure!(
                ["system", "developer", "user", "assistant", "tool"].contains(&role.as_str()),
                "unsupported message role {role}"
            );
            if role == "developer" {
                role = "system".into();
                m.insert("role".into(), json!(role));
            }
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                None | Some(Value::Null) if role == "assistant" => String::new(),
                Some(Value::Array(parts)) => {
                    let mut s = String::new();
                    for part in parts {
                        ensure!(
                            part["type"].as_str() == Some("text"),
                            "native chat supports text content only"
                        );
                        s.push_str(part["text"].as_str().context("text part requires text")?);
                    }
                    s
                }
                _ => bail!("message content must be text"),
            };
            m.insert("content".into(), json!(content));
            if let Some(calls) = m.get_mut("tool_calls") {
                ensure!(role == "assistant", "tool_calls require assistant role");
                let calls = calls
                    .as_array_mut()
                    .context("tool_calls must be an array")?;
                for call in calls {
                    ensure!(
                        call["type"].as_str() == Some("function"),
                        "only function tool calls are supported"
                    );
                    let id = call["id"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .context("tool call id is required")?
                        .to_owned();
                    let name = call["function"]["name"]
                        .as_str()
                        .filter(|s| name_ok(s))
                        .context("invalid tool function name")?
                        .to_owned();
                    ensure!(pending.insert(id, name).is_none(), "duplicate tool call ID");
                    let args = &mut call["function"]["arguments"];
                    if let Some(s) = args.as_str() {
                        *args =
                            serde_json::from_str(s).context("tool arguments must contain JSON")?;
                    }
                    ensure!(args.is_object(), "tool arguments must be a JSON object");
                }
            }
            if role == "tool" {
                let id = m
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .context("tool result requires tool_call_id")?;
                let name = pending
                    .remove(id)
                    .context("tool result does not match a preceding tool call")?;
                m.insert("name".into(), json!(name));
            }
            if role == "system"
                && messages
                    .last()
                    .is_some_and(|m: &Value| m["role"] == "system")
            {
                let last = messages.last_mut().unwrap();
                let prior = last["content"].as_str().unwrap();
                last["content"] = json!(format!("{prior}\n\n{content}"));
            } else {
                messages.push(Value::Object(m));
            }
        }
        ensure!(
            pending.is_empty(),
            "all assistant tool calls need matching tool results before generation"
        );
        ensure!(
            messages
                .last()
                .is_some_and(|m| m["role"] == "user" || m["role"] == "tool"),
            "generation requires a final user or tool message"
        );
        let mut tools = match v.get("tools") {
            None => vec![],
            Some(x) => x.as_array().context("tools must be an array")?.clone(),
        };
        ensure!(tools.len() <= 128, "too many tools");
        let mut names = BTreeSet::new();
        for t in &tools {
            ensure!(t["type"] == "function", "only function tools are supported");
            let name = t["function"]["name"]
                .as_str()
                .filter(|s| name_ok(s))
                .context("invalid function name")?;
            ensure!(names.insert(name), "duplicate tool name");
            ensure!(
                t["function"]["parameters"].is_object(),
                "function parameters must be a schema object"
            );
            ensure!(
                t["function"].get("strict").is_none_or(|x| x == false),
                "strict tool-schema constrained decoding is not implemented"
            );
        }
        if let Some(choice) = v.get("tool_choice") {
            match choice.as_str() {
                Some("auto") => {}
                Some("none") => tools.clear(),
                _ => bail!(
                    "tool_choice supports auto or none; forced/constrained calls are not implemented"
                ),
            }
        }
        let reserve = match (v.get("max_tokens"), v.get("max_completion_tokens")) {
            (Some(a), Some(b)) => {
                ensure!(a == b, "max_tokens and max_completion_tokens disagree");
                a.as_u64()
                    .context("token limit must be a positive integer")?
            }
            (Some(a), None) | (None, Some(a)) => a
                .as_u64()
                .context("token limit must be a positive integer")?,
            _ => max_output.min(1024) as u64,
        };
        let max_tokens = usize::try_from(reserve)?;
        ensure!(
            max_tokens > 0 && max_tokens <= max_output,
            "token limit must be in 1..={max_output}"
        );
        let mut stops = vec![];
        if let Some(x) = v.get("stop") {
            match x {
                Value::Null => {}
                Value::String(s) => stops.push(s.clone()),
                Value::Array(a) => {
                    for x in a {
                        stops.push(x.as_str().context("stop entries must be strings")?.into());
                    }
                }
                _ => bail!("stop must be a string or array"),
            }
        }
        ensure!(
            stops.len() <= 16 && stops.iter().all(|s| !s.is_empty() && s.len() <= 256),
            "stop supports up to16 nonempty strings of at most256 bytes"
        );
        let opts = v.get("stream_options");
        if let Some(x) = opts {
            ensure!(
                x.is_object() && x.as_object().unwrap().keys().all(|k| k == "include_usage"),
                "unsupported stream_options"
            );
        }
        let kwargs = v.get("chat_template_kwargs");
        if let Some(x) = kwargs {
            ensure!(
                x.is_object()
                    && x.as_object()
                        .unwrap()
                        .keys()
                        .all(|k| k == "enable_thinking" || k == "preserve_thinking"),
                "unsupported chat_template_kwargs"
            );
        }
        if let Some(x) = kwargs.and_then(|x| x.get("preserve_thinking")) {
            ensure!(
                x == true,
                "native API preserves prior reasoning; preserve_thinking must be true"
            );
        }
        Ok(Self {
            model,
            messages,
            tools,
            max_tokens,
            stream: bool_value(v, "stream", false)?,
            include_usage: opts
                .map(|v| bool_value(v, "include_usage", false))
                .transpose()?
                .unwrap_or(false),
            thinking: kwargs
                .map(|v| bool_value(v, "enable_thinking", true))
                .transpose()?
                .unwrap_or(true),
            opens_reasoning: false,
            stops,
            temperature: float_value(v, "temperature", 0.0, 0.0, 2.0)?,
            top_p: float_value(v, "top_p", 1.0, 0.000001, 1.0)?,
            seed: v
                .get("seed")
                .map(|x| x.as_u64().context("seed must be a nonnegative integer"))
                .transpose()?
                .unwrap_or(0x9070),
            frequency_penalty: float_value(v, "frequency_penalty", 0.0, -2.0, 2.0)?,
            presence_penalty: float_value(v, "presence_penalty", 0.0, -2.0, 2.0)?,
            parallel_tool_calls: bool_value(v, "parallel_tool_calls", true)?,
        })
    }
}

pub struct Sampler {
    state: u64,
    counts: BTreeMap<u32, usize>,
    scratch: Vec<(u32, f64)>,
}
impl Sampler {
    pub fn new(seed: u64, prompt: &[u32]) -> Self {
        let mut s = Self {
            state: seed,
            counts: BTreeMap::new(),
            scratch: vec![],
        };
        for &t in prompt {
            *s.counts.entry(t).or_default() += 1;
        }
        s
    }
    pub fn sample(&mut self, logits: &[f32], r: &ChatRequest) -> Result<u32> {
        ensure!(
            !logits.is_empty() && logits.iter().all(|x| x.is_finite()),
            "non-finite/empty model logits"
        );
        if r.temperature == 0.0 && r.frequency_penalty == 0.0 && r.presence_penalty == 0.0 {
            let token = crate::runtime::greedy(logits)?;
            *self.counts.entry(token).or_default() += 1;
            return Ok(token);
        }
        self.scratch.clear();
        self.scratch.extend(
            logits
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u32, *v as f64)),
        );
        if r.frequency_penalty != 0.0 || r.presence_penalty != 0.0 {
            for (&token, &n) in &self.counts {
                if let Some((_, score)) = self.scratch.get_mut(token as usize) {
                    *score -= r.frequency_penalty as f64 * n as f64 + r.presence_penalty as f64;
                }
            }
        }
        let best = self
            .scratch
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.1.total_cmp(&b.1).then(ib.cmp(ia)))
            .unwrap()
            .0;
        let token = if r.temperature == 0.0 {
            self.scratch[best].0
        } else {
            let maximum = self.scratch[best].1;
            let mut sum = 0.0;
            if r.top_p < 1.0 {
                self.scratch
                    .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            }
            for (_, p) in &mut self.scratch {
                *p = ((*p - maximum) / r.temperature as f64).exp();
                sum += *p;
            }
            let mut retained = 0.0;
            let mut end = 0;
            for (_, p) in &self.scratch {
                retained += *p;
                end += 1;
                if retained >= sum * r.top_p as f64 {
                    break;
                }
            }
            self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let mut pick = ((z >> 11) as f64) * (1.0 / (1u64 << 53) as f64) * retained;
            let mut selected = self.scratch[end - 1].0;
            for &(id, p) in &self.scratch[..end] {
                if pick < p {
                    selected = id;
                    break;
                }
                pick -= p;
            }
            selected
        };
        *self.counts.entry(token).or_default() += 1;
        Ok(token)
    }
}

// Tool markup in a Markdown example is text, not a request to execute a tool.
// Scan only the unconsumed answer: an actual call's JSON/XML body is handled by
// the protocol parser and must not change the Markdown state of later content.
fn next_tool_call(text: &str, mut offset: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut fence: Option<(u8, usize)> = None;
    while offset < bytes.len() {
        if offset == 0 || bytes[offset - 1] == b'\n' {
            let line_end = bytes[offset..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(bytes.len(), |n| offset + n);
            let mut marker = offset;
            while marker < line_end && bytes[marker] == b' ' && marker - offset < 4 {
                marker += 1;
            }
            if let Some((open, size)) = fence {
                if marker - offset <= 3 && marker < line_end && bytes[marker] == open {
                    let mut end = marker + 1;
                    while end < line_end && bytes[end] == open {
                        end += 1;
                    }
                    if end - marker >= size
                        && bytes[end..line_end]
                            .iter()
                            .all(|b| matches!(b, b' ' | b'\t' | b'\r'))
                    {
                        fence = None;
                    }
                }
                // Consume the closing line too; its delimiter must not open
                // an inline span that hides a later, real function call.
                offset = line_end + usize::from(line_end < bytes.len());
                continue;
            }
            if marker - offset <= 3
                && marker < line_end
                && (bytes[marker] == b'`' || bytes[marker] == b'~')
            {
                let ch = bytes[marker];
                let mut end = marker + 1;
                while end < line_end && bytes[end] == ch {
                    end += 1;
                }
                let count = end - marker;
                if count >= 3 && (ch != b'`' || !bytes[end..line_end].contains(&b'`')) {
                    fence = Some((ch, count));
                    offset = line_end + usize::from(line_end < bytes.len());
                    continue;
                }
            }
        }
        // A backslash-escaped delimiter stays literal. Backslashes inside an
        // already-open code span have no special meaning.
        if bytes[offset] == b'\\'
            && bytes
                .get(offset + 1)
                .is_some_and(|b| matches!(b, b'`' | b'<' | b'\\'))
        {
            offset += 2;
            continue;
        }
        if bytes[offset] == b'`' {
            let mut end = offset + 1;
            while end < bytes.len() && bytes[end] == b'`' {
                end += 1;
            }
            let width = end - offset;
            let mut candidate = end;
            let mut closed = None;
            while candidate < bytes.len() {
                if bytes[candidate] != b'`' {
                    candidate += 1;
                    continue;
                }
                let mut next = candidate + 1;
                while next < bytes.len() && bytes[next] == b'`' {
                    next += 1;
                }
                if next - candidate == width {
                    closed = Some(next);
                    break;
                }
                candidate = next;
            }
            // Unmatched backticks are literal Markdown; do not hide an actual
            // tool call later in the answer because of one incomplete span.
            offset = closed.unwrap_or(end);
            continue;
        }
        if bytes[offset..].starts_with(b"<tool_call>") {
            return Some(offset);
        }
        offset += 1;
    }
    None
}

/// Return visible text, optional reasoning and validated OpenAI function calls.
pub fn parse_output(text: &str, request: &ChatRequest, id: &str) -> Result<Value> {
    let (reasoning, answer) = if request.opens_reasoning || text.starts_with("<think>") {
        if let Some((r, a)) = text.split_once("</think>") {
            (
                Some(r.trim_start_matches("<think>").trim().to_owned()),
                a.trim_start_matches('\n'),
            )
        } else {
            (Some(text.trim_start_matches("<think>").to_owned()), "")
        }
    } else {
        (None, text)
    };
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut consumed = 0;
    while !request.tools.is_empty() {
        let Some(start) = next_tool_call(answer, consumed) else {
            break;
        };
        content.push_str(&answer[consumed..start]);
        let body = &answer[start + 11..];
        let Some(end) = body.find("</tool_call>") else {
            consumed = start;
            break;
        };
        let body = body[..end].trim();
        let (name, args) = if body.starts_with('{') {
            let call: Value =
                serde_json::from_str(body).context("model emitted invalid tool-call JSON")?;
            let name = call["name"]
                .as_str()
                .context("model tool call lacks name")?
                .to_owned();
            let mut args = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
            if let Some(s) = args.as_str() {
                args = serde_json::from_str(s)?;
            }
            (name, args)
        } else {
            let s = body
                .strip_prefix("<function=")
                .context("unsupported generated tool-call format")?;
            let n = s.find('>').context("unterminated function name")?;
            let name = s[..n].to_owned();
            let mut parameters = s[n + 1..].trim();
            parameters = parameters
                .strip_suffix("</function>")
                .context("unterminated function call")?
                .trim();
            let schema = request
                .tools
                .iter()
                .find(|t| t["function"]["name"] == name)
                .map(|t| &t["function"]["parameters"]["properties"]);
            let mut args = serde_json::Map::new();
            while !parameters.is_empty() {
                let p = parameters
                    .strip_prefix("<parameter=")
                    .context("invalid parameter tag")?;
                let n = p.find('>').context("unterminated parameter name")?;
                let key = &p[..n];
                let p = &p[n + 1..];
                let end = p.find("</parameter>").context("unterminated parameter")?;
                let raw = p[..end]
                    .strip_prefix('\n')
                    .unwrap_or(&p[..end])
                    .strip_suffix('\n')
                    .unwrap_or_else(|| p[..end].strip_prefix('\n').unwrap_or(&p[..end]));
                let value = if schema.is_some_and(|s| s[key]["type"] == "string") {
                    json!(raw)
                } else {
                    serde_json::from_str(raw).unwrap_or_else(|_| json!(raw))
                };
                ensure!(
                    args.insert(key.to_owned(), value).is_none(),
                    "duplicate model tool argument"
                );
                parameters = p[end + 12..].trim();
            }
            (name, Value::Object(args))
        };
        ensure!(
            request.tools.iter().any(|t| t["function"]["name"] == name),
            "model called an unavailable function: {name}"
        );
        ensure!(
            args.is_object(),
            "generated arguments must be a JSON object"
        );
        calls.push(json!({"id":format!("call_{id}_{}",calls.len()),"type":"function","function":{"name":name,"arguments":serde_json::to_string(&args)?}}));
        consumed = start + 11 + end + 12;
    }
    content.push_str(&answer[consumed..]);
    ensure!(
        request.parallel_tool_calls || calls.len() <= 1,
        "model emitted multiple tool calls while parallel_tool_calls=false"
    );
    let mut message = json!({"role":"assistant","content":content});
    if let Some(r) = reasoning {
        message["reasoning_content"] = json!(r);
    }
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn req() -> ChatRequest {
        ChatRequest::parse(&json!({"model":"m","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false},"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}),128).unwrap()
    }
    #[test]
    fn template_json_keeps_utf8_html_and_python_spacing() {
        let mut env = minijinja::Environment::new();
        env.add_filter("tojson", template_json);
        let rendered = env
            .render_str(
                "{{ x|tojson }}",
                minijinja::value::Serde(serde_json::json!({"x":{"a":"café <>&","b":[1,true]}})),
            )
            .unwrap();
        assert_eq!(rendered, "{\"a\": \"café <>&\", \"b\": [1, true]}");
    }
    #[test]
    #[ignore = "requires NEXTORA_TEMPLATE_PARITY_JSON with externally generated Python-Jinja reference cases"]
    fn external_template_byte_parity() {
        let path =
            std::env::var("NEXTORA_TEMPLATE_PARITY_JSON").expect("set NEXTORA_TEMPLATE_PARITY_JSON");
        let data: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut g = Gguf::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-qwen3moe.gguf"),
        )
        .unwrap();
        g.meta.insert(
            "tokenizer.chat_template".into(),
            crate::gguf::Value::String(data["template"].as_str().unwrap().into()),
        );
        let template = ChatTemplate::new(&g).unwrap();
        let cases = data["cases"].as_array().unwrap();
        assert!(cases.len() >= 5);
        for case in cases {
            let request = ChatRequest::parse(&case["request"], 4096).unwrap();
            assert_eq!(
                template.render(&request).unwrap(),
                case["expected"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
    #[test]
    fn native_tool_json_xml_and_reasoning() {
        let r = req();
        for t in [
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.rs\"}}</tool_call>",
            "<tool_call><function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function></tool_call>",
        ] {
            let o = parse_output(t, &r, "x").unwrap();
            assert_eq!(
                o["tool_calls"][0]["function"]["arguments"],
                "{\"path\":\"a.rs\"}"
            );
        }
        let o = parse_output("<think>plan</think>\nanswer", &r, "x").unwrap();
        assert_eq!(o["reasoning_content"], "plan");
        assert_eq!(o["content"], "answer");
    }
    #[test]
    fn literal_tool_markup_without_enabled_tools() {
        let mut r = req();
        r.tools.clear();
        for text in [
            "let marker = \"</think><tool_call>example</tool_call>\";",
            "<tool_call>{\"name\":\"unavailable\",\"arguments\":{}}</tool_call>",
            "<tool_call>invalid JSON or XML</tool_call>",
            "<tool_call>an unfinished example",
        ] {
            let output = parse_output(text, &r, "literal").unwrap();
            assert_eq!(output["content"], text);
            assert!(output.get("tool_calls").is_none());
            assert!(output.get("reasoning_content").is_none());
        }
        let r = ChatRequest::parse(
            &json!({"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":req().tools,"tool_choice":"none"}),
            128,
        )
        .unwrap();
        let text = "<tool_call>{\"name\":\"read_file\",\"arguments\":{}}</tool_call>";
        let output = parse_output(text, &r, "disabled").unwrap();
        assert_eq!(output["content"], text);
        assert!(output.get("tool_calls").is_none());
    }
    #[test]
    fn markdown_examples_preserve_literals_before_real_tool_calls() {
        let r = req();
        let real =
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.rs\"}}</tool_call>";
        let examples = [
            "Example:\n```xml\n<tool_call>not a function</tool_call>\n```\n",
            "Example:\n~~~xml\n<tool_call>not a function</tool_call>\n~~~\n",
            "````md\n```xml\n<tool_call>literal</tool_call>\n```\n````\n",
            "   ```xml\r\n<tool_call>literal</tool_call>\r\n   ````\t\r\n",
            "Use `<tool_call>literal</tool_call>` in the docs.\n",
            "Use ``a ` <tool_call>literal</tool_call>`` in the docs.\n",
            "λ: `<tool_call>literal</tool_call>` → example\n",
            "\\<tool_call>escaped example</tool_call>\n",
        ];
        for example in examples {
            let output = parse_output(&format!("{example}{real}"), &r, "example").unwrap();
            assert_eq!(output["content"], example, "{example:?}");
            assert_eq!(output["tool_calls"].as_array().unwrap().len(), 1);
            assert_eq!(output["tool_calls"][0]["function"]["name"], "read_file");
        }
    }
    #[test]
    fn markdown_fences_only_close_with_matching_delimiters() {
        let r = req();
        for text in [
            "```xml\n<tool_call>literal</tool_call>",
            "~~~~xml\n~~~\n<tool_call>literal</tool_call>\n~~~~ trailing text\n",
            "```xml\n~~~\n<tool_call>literal</tool_call>\n``` trailing text\n",
        ] {
            let output = parse_output(text, &r, "unfinished").unwrap();
            assert_eq!(output["content"], text);
            assert!(output.get("tool_calls").is_none());
        }
        let text =
            "a stray ` then <tool_call>{\"name\":\"read_file\",\"arguments\":{}}</tool_call>";
        let output = parse_output(text, &r, "unmatched").unwrap();
        assert_eq!(output["content"], "a stray ` then ");
        assert_eq!(output["tool_calls"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn real_tool_call_does_not_change_later_markdown_state() {
        let r = req();
        let real =
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"`a.rs\"}}</tool_call>";
        let example = "\n```xml\n<tool_call>literal</tool_call>\n```\n";
        let output = parse_output(&format!("{real}{example}{real}"), &r, "two").unwrap();
        assert_eq!(output["content"], example);
        assert_eq!(output["tool_calls"].as_array().unwrap().len(), 2);
        assert_ne!(output["tool_calls"][0]["id"], output["tool_calls"][1]["id"]);
    }
    #[test]
    fn default_thinking_does_not_hide_plain_or_literal_code_answers() {
        let r = ChatRequest::parse(
            &json!({"model":"m","messages":[{"role":"user","content":"hi"}]}),
            128,
        )
        .unwrap();
        assert!(r.thinking);
        assert!(!r.opens_reasoning);
        for text in [
            "plain answer",
            "let close = \"</think>\";",
            "Use <think> and </think> markers.",
        ] {
            let output = parse_output(text, &r, "reasoning").unwrap();
            assert_eq!(output["content"], text);
            assert!(output.get("reasoning_content").is_none());
        }
        let output = parse_output("<think>plan</think>\nanswer", &r, "generated").unwrap();
        assert_eq!(output["reasoning_content"], "plan");
        assert_eq!(output["content"], "answer");
    }
    #[test]
    fn actual_reasoning_prefix_controls_reasoning_extraction() {
        let mut r = req();
        r.opens_reasoning = true;
        r.tools.clear();
        let output = parse_output("plan</think>\nlet close = \"</think>\";", &r, "prefix").unwrap();
        assert_eq!(output["reasoning_content"], "plan");
        assert_eq!(output["content"], "let close = \"</think>\";");
        let output = parse_output("unfinished plan", &r, "truncated").unwrap();
        assert_eq!(output["reasoning_content"], "unfinished plan");
        assert_eq!(output["content"], "");
    }
    #[test]
    fn generation_annotation_keeps_whitespace_control() {
        let x =
            without_generation_annotations("a {%- generation %}x{% endgeneration -%} z").unwrap();
        let e = minijinja::Environment::new();
        assert_eq!(e.render_str(&x, ()).unwrap(), "axz");
        assert!(without_generation_annotations("{% endgeneration %}").is_err());
    }
    #[test]
    fn request_validation_and_tool_identity() {
        let mut v =
            json!({"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":-1});
        assert!(ChatRequest::parse(&v, 128).is_err());
        v["temperature"] = json!(0);
        v["messages"] = json!([{"role":"tool","tool_call_id":"wrong","content":"x"}]);
        assert!(ChatRequest::parse(&v, 128).is_err());
        v["messages"] = json!([{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"x"}]);
        assert!(ChatRequest::parse(&v, 128).is_ok());
    }
    #[test]
    fn sampler_stable_seed_and_finite_gate() {
        let mut r = req();
        r.temperature = 1.0;
        let mut a = Sampler::new(123, &[]);
        let mut b = Sampler::new(123, &[]);
        for _ in 0..100 {
            assert_eq!(
                a.sample(&[0., 1., 2.], &r).unwrap(),
                b.sample(&[0., 1., 2.], &r).unwrap()
            );
        }
        assert!(a.sample(&[f32::NAN], &r).is_err());
    }
}
