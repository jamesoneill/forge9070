//! Forge9070 model-management regression tests. These execute the real CPU
//! engine with original synthetic weights; no network model download is needed.
use rusting_strata::{gguf::Gguf, serve::list_models};
use serde_json::{Value, json};
use std::{io::{Read, Write}, net::{TcpListener, TcpStream}, path::{Path, PathBuf},
    process::{Child, Command, Stdio}, time::{Duration, Instant}};

struct Server(Child, PathBuf);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = std::fs::remove_dir_all(&self.1);
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf")
}

fn request(addr: &str, path: &str, body: Option<Value>, authorized: bool) -> (u16, Value) {
    let mut stream = TcpStream::connect(addr).expect("connect local server");
    stream.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    let method = if body.is_some() { "POST" } else { "GET" };
    let body = body.map(|v| v.to_string()).unwrap_or_default();
    let key = if authorized { "Authorization: Bearer forge9070-test-admin\r\n" } else { "" };
    write!(stream, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\n{key}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut text = String::new();
    stream.read_to_string(&mut text).unwrap();
    let (headers, body) = text.split_once("\r\n\r\n").expect("HTTP response");
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).expect(body))
}

fn chat(addr: &str, model: &str) -> Value {
    let (status, value) = request(addr, "/v1/chat/completions", Some(json!({
        "model": model, "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}
    })), true);
    assert_eq!(status, 200, "{value}");
    value
}

#[test]
fn real_cpu_models_switch_keep_invalid_loads_out_and_unload() {
    let root = std::env::temp_dir().join(format!("forge9070-switch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let a_id = "hf/owner-a/commit-a/nested/weights.gguf";
    let b_id = "hf/owner-b/commit-b/nested/weights.gguf";
    for id in [a_id, b_id] { std::fs::create_dir_all(root.join(id).parent().unwrap()).unwrap(); }
    std::fs::copy(fixture(), root.join(a_id)).unwrap();

    // Same architecture/tokenizer, distinct real output projection: B replaces A.
    let g = Gguf::open(&fixture()).unwrap();
    let output = &g.tensors["output.weight"];
    let data = g.bytes(output);
    let mut bytes = std::fs::read(fixture()).unwrap();
    let offset = bytes.windows(data.len()).position(|window| window == data).unwrap();
    let row = output.dims[0] as usize * 4;
    bytes[offset + 65 * row..offset + 66 * row].fill(0);
    for f in bytes[offset + 66 * row..offset + 67 * row].chunks_exact_mut(4) {
        f.copy_from_slice(&1.0_f32.to_le_bytes());
    }
    std::fs::write(root.join(b_id), bytes).unwrap();
    std::fs::write(root.join("broken.gguf"), b"GGUFnot a valid checkpoint").unwrap();
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3.gguf"), root.join("dense.gguf")).unwrap();

    let addr = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
    let child = Command::new(env!("CARGO_BIN_EXE_rustingstrata"))
        .args(["serve", "--addr", &addr, "--ctx", "128", "--max-tokens", "16", "--ram-margin", "0", "--api-key", "forge9070-test-admin"])
        .arg("--models-dir").arg(&root).env_remove("STRATA_API_KEY")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let mut server = Server(child, root);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        assert!(server.0.try_wait().unwrap().is_none(), "server exited during startup");
        if TcpStream::connect(&addr).is_ok() { break; }
        assert!(Instant::now() < deadline, "server startup timed out");
        std::thread::sleep(Duration::from_millis(25));
    }
    let (status, no_model) = request(&addr, "/v1/models", None, true);
    assert_eq!(status, 200);
    assert_eq!(no_model["data"], json!([]));
    assert_eq!(request(&addr, "/api/load", Some(json!({"model": a_id})), false).0, 401);
    let (status, catalog) = request(&addr, "/api/models", None, true);
    assert_eq!(status, 200);
    assert!(catalog["models"].as_array().unwrap().iter().any(|m| m["id"] == a_id));
    assert!(catalog["models"].as_array().unwrap().iter().any(|m| m["id"] == b_id));
    assert!(!catalog["models"].as_array().unwrap().iter().any(|m| m["id"] == "dense.gguf"));
    assert_eq!(catalog["policy"], "moe_only");

    let (status, loaded) = request(&addr, "/api/load", Some(json!({"model": a_id})), true);
    assert_eq!(status, 200, "{loaded}");
    assert_eq!(loaded["loaded"], a_id);
    assert_eq!(chat(&addr, a_id)["choices"][0]["message"]["content"], "AAAA");
    assert_eq!(request(&addr, "/api/load", Some(json!({"model": "../outside.gguf"})), true).0, 404);
    assert_eq!(request(&addr, "/api/load", Some(json!({"model": "broken.gguf"})), true).0, 400);
    assert_eq!(request(&addr, "/api/load", Some(json!({"model": "dense.gguf"})), true).0, 400);
    assert_eq!(request(&addr, "/v1/models", None, true).1["data"][0]["id"], a_id);
    assert_eq!(chat(&addr, a_id)["choices"][0]["message"]["content"], "AAAA");

    let (status, loaded) = request(&addr, "/api/load", Some(json!({"model": b_id})), true);
    assert_eq!(status, 200, "{loaded}");
    assert_eq!(loaded["loaded"], b_id);
    assert_eq!(request(&addr, "/v1/models", None, true).1["data"][0]["id"], b_id);
    assert_eq!(chat(&addr, b_id)["choices"][0]["message"]["content"], "BBBB");
    let (status, stale) = request(&addr, "/v1/chat/completions", Some(json!({
        "model": a_id, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4
    })), true);
    assert_eq!(status, 409, "a stale explicit model ID must not silently use B");
    assert_eq!(stale["error"]["loaded_model"], b_id);
    let (status, unloaded) = request(&addr, "/api/unload", Some(json!({})), true);
    assert_eq!(status, 200);
    assert_eq!(unloaded["unloaded"], true);
    assert_eq!(request(&addr, "/v1/models", None, true).1["data"], json!([]));
    assert_eq!(request(&addr, "/health", None, false).0, 503);
    assert_eq!(request(&addr, "/api/load", Some(json!({"model": a_id})), true).0, 200);
    assert_eq!(chat(&addr, a_id)["choices"][0]["message"]["content"], "AAAA");
}

#[test]
fn recursive_catalog_groups_shards_and_never_follows_symlinks() {
    let root = std::env::temp_dir().join(format!("forge9070-catalog-{}", std::process::id()));
    let outside = root.with_extension("outside");
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&outside);
    std::fs::create_dir_all(root.join("hf/org/commit/subdir")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    for (name, n) in [("m-00001-of-00002.gguf", 5), ("m-00002-of-00002.gguf", 7), ("part.gguf.part", 10)] {
        std::fs::write(root.join("hf/org/commit/subdir").join(name), vec![0; n]).unwrap();
    }
    std::fs::write(outside.join("escape.gguf"), b"outside").unwrap();
    #[cfg(unix)] {
        std::os::unix::fs::symlink(&outside, root.join("external")).unwrap();
        std::os::unix::fs::symlink(outside.join("escape.gguf"), root.join("escaped.gguf")).unwrap();
        std::os::unix::fs::symlink(&root, root.join("cycle")).unwrap();
    }
    let models = list_models(&root);
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].0, "hf/org/commit/subdir/m-00001-of-00002.gguf");
    assert_eq!(models[0].2, 12);
    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(outside).unwrap();
}
