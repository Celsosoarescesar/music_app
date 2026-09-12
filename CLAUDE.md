# music_app

## Project Overview

API FastAPI de teste para o modelo de geração de música ACE-Step
(https://github.com/ace-step/ACE-Step). Projeto de aprendizado: o objetivo
é aprender a operar o modelo (não é um produto). Roda em dois modos:

- `MUSIC_BACKEND=mock` (padrão): gera áudio sintético, sem GPU, sem o
  pacote `acestep` instalado. Usado no desenvolvimento local e nos testes.
- `MUSIC_BACKEND=acestep`: usa o modelo real. Requer GPU — pensado para
  rodar dentro de um notebook Kaggle (ver `notebooks/`).

## Tech Stack

- Python >=3.10, gerenciado com `uv`
- FastAPI + Uvicorn
- Pydantic v2 / pydantic-settings
- NumPy + soundfile (codificação/decodificação de WAV)
- pytest + httpx (testes)
- Opcional (grupo `kaggle`): `acestep`, `pyngrok`

## Project Structure

```
app/
├── main.py                # rotas FastAPI, lifespan, middleware de log
├── settings.py             # AppSettings (pydantic-settings)
├── schemas.py               # Pydantic request models por endpoint
├── backends/
│   ├── base.py               # Protocol MusicBackend (6 métodos)
│   ├── mock_backend.py       # backend sintético (sem GPU)
│   └── acestep_backend.py    # backend real (Kaggle)
└── utils.py                 # conversão array de áudio <-> buffer WAV
tests/                       # pytest, sempre roda com MUSIC_BACKEND=mock
notebooks/                   # notebook para subir a API real no Kaggle
docs/superpowers/            # spec e plano de implementação deste projeto
```

## Commands

- `uv sync` — instala as dependências base (modo mock/dev).
- `uv sync --extra kaggle` — inclui também `acestep` e `pyngrok` (só dentro
  do Kaggle).
- `uv run pytest -v` — roda a suíte de testes (sempre em modo mock).
- `uv run uvicorn app.main:app --reload` — sobe a API localmente em modo
  mock (padrão) em `http://127.0.0.1:8000`.
- `MUSIC_BACKEND=acestep uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`
  — sobe a API com o modelo real (requer GPU e `acestep` instalado).

## Endpoints (todos síncronos, retornam `audio/wav`)

`POST /generate/text2music`, `/generate/retake`, `/generate/repaint`,
`/generate/edit`, `/generate/extend`, `/generate/audio2audio`. Veja
`docs/superpowers/specs/2026-09-12-music-app-ace-step-design.md` para os
campos de cada um.

## Known limitations (by design)

- Tom/BPM não são parâmetros estruturados — vão como texto livre em `tags`.
- Sem separação de stems (StemGen do ACE-Step ainda não foi lançado).
- Sem voice cloning (ambíguo entre inferência e treino de LoRA — fora do
  v1).
