Write-Host "[*] Initialisation du projet Zilda sur Windows..." -ForegroundColor Cyan

# 1. Création de la structure des répertoires
New-Item -ItemType Directory -Force -Path "zilda-core/src/api", "zilda-core/src/orchestrator", "zilda-core/src/memory", "zilda-core/src/backend" | Out-Null
New-Item -ItemType Directory -Force -Path "zilda-python/src", "zilda-python/zilda" | Out-Null
New-Item -ItemType Directory -Force -Path "zilda-js/src" | Out-Null
New-Item -ItemType Directory -Force -Path "examples" | Out-Null

# 2. Fichiers racine
@'
[workspace]
members = [
    "zilda-core",
    "zilda-python"
]
resolver = "2"
'@ | Set-Content -Path "Cargo.toml" -Encoding utf8

@'
.PHONY: build test clean

build:
	cargo build --release

test:
	cargo test

clean:
	cargo clean
'@ | Set-Content -Path "Makefile" -Encoding utf8

"# Zilda" | Set-Content -Path "README.md" -Encoding utf8

# 3. Configuration zilda-core
@'
[package]
name = "zilda-core"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
candle-core = { version = "0.8" }
axum = "0.7"
'@ | Set-Content -Path "zilda-core/Cargo.toml" -Encoding utf8

New-Item -ItemType File -Force -Path "zilda-core/src/main.rs", "zilda-core/src/api/mod.rs", "zilda-core/src/orchestrator/mod.rs", "zilda-core/src/memory/mod.rs", "zilda-core/src/backend/mod.rs" | Out-Null

# 4. Configuration zilda-python
@'
[package]
name = "zilda-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "zilda_bindings"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
zilda-core = { path = "../zilda-core" }
'@ | Set-Content -Path "zilda-python/Cargo.toml" -Encoding utf8

@'
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "zilda"
version = "0.1.0"
requires-python = ">=3.8"
'@ | Set-Content -Path "zilda-python/pyproject.toml" -Encoding utf8

New-Item -ItemType File -Force -Path "zilda-python/src/lib.rs", "zilda-python/zilda/__init__.py", "zilda-python/zilda/inference.py" | Out-Null

# 5. Configuration zilda-js
@'
{
  "name": "zilda",
  "version": "0.1.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc"
  },
  "dependencies": {
    "napi-rs": "^2.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}
'@ | Set-Content -Path "zilda-js/package.json" -Encoding utf8

New-Item -ItemType File -Force -Path "zilda-js/src/lib.rs", "zilda-js/index.ts" | Out-Null

# 6. Exemples
New-Item -ItemType File -Force -Path "examples/production_server.py", "examples/pipeline_moe.ts" | Out-Null

Write-Host "[+] Structure Zilda créée avec succès pour l'environnement Windows." -ForegroundColor Green