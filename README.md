# music_app

Projeto de teste para aprender a usar o modelo de geração de música
[ACE-Step](https://github.com/ace-step/ACE-Step) através de uma API
FastAPI.

## Rodando localmente (modo mock, sem GPU)

```bash
uv sync
uv run pytest -v
uv run uvicorn app.main:app --reload
```

Testando um endpoint (áudio sintético, só para validar o contrato da API):

```bash
curl -X POST http://127.0.0.1:8000/generate/text2music \
  -H "Content-Type: application/json" \
  -d '{"tags": "lo-fi, chill, piano", "lyrics": "", "duration": 10}' \
  --output saida.wav
```

## Rodando no Kaggle (modelo real, com GPU)

1. Faça upload de `notebooks/ace_step_api_kaggle.ipynb` num notebook Kaggle
   com acelerador de GPU ativado (T4 ou P100).
2. Rode as células na ordem — elas instalam as dependências (incluindo o
   `acestep`), sobem a API com `MUSIC_BACKEND=acestep` e abrem um túnel
   `ngrok` público.
3. Use a URL impressa pelo notebook para chamar os mesmos endpoints de
   fora do Kaggle (ex: com `curl` ou o cliente HTTP que preferir).

A sessão do Kaggle expira depois de algumas horas e a URL do `ngrok` muda a
cada nova sessão (plano gratuito) — normal para um projeto de teste.

## Documentação

- Spec de design: `docs/superpowers/specs/2026-09-12-music-app-ace-step-design.md`
- Plano de implementação: `docs/superpowers/plans/2026-09-12-music-app-ace-step-plan.md`
