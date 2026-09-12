# music_app — API FastAPI de teste para o modelo ACE-Step

Data: 2026-09-12
Status: aprovado, aguardando plano de implementação

## 1. Objetivo

Projeto de teste/aprendizado para usar o modelo de geração de música
[ACE-Step](https://github.com/ace-step/ACE-Step) (Apache 2.0) através de uma
API FastAPI. O objetivo não é produto — é aprender a integrar e operar o
modelo de ponta a ponta (API real chamando o modelo real, rodando com GPU).

Referências usadas para desenhar o projeto:

- `C:\estudos\aprender_cc` — materiais de curso sobre organizar projetos com
  Claude Code. Padrão adotado: `CLAUDE.md` na raiz (visão geral, stack,
  estrutura, comandos), separação clara `app/` (código) / `tests/`.
- `C:\estudos\base\building-generative-ai-services-main\app` — referência de
  API FastAPI servindo modelos generativos (texto/áudio/imagem/vídeo/3D).
  Padrão adotado: `main.py` (rotas + `lifespan` que carrega o modelo uma vez +
  middleware de log), `models.py`/backends (load/generate), `schemas.py`
  (Pydantic), `settings.py` (pydantic-settings), `utils.py` (helpers de
  conversão de buffer). Respostas binárias via `StreamingResponse`.

## 2. Onde o modelo roda (decisão-chave)

O ACE-Step precisa de GPU (mínimo recomendado 8GB VRAM com
`cpu_offload=true`). O ambiente local de desenvolvimento não tem GPU
dedicada para isso; o plano é usar o Kaggle (notebooks com GPU gratuita:
T4/P100, 16GB VRAM, suficientes).

**Decisão: modo híbrido.** Uma única base de código FastAPI roda em dois
modos, escolhidos por variável de ambiente `MUSIC_BACKEND`:

- `mock` (padrão, para dev local sem GPU): gera áudio sintético (tom
  senoidal) com a duração/forma correta, só para validar o contrato da API
  e o fluxo ponta a ponta sem depender de GPU nem do pacote `acestep`
  instalado.
- `acestep` (usado dentro do notebook Kaggle): carrega o pacote `acestep` de
  verdade e gera música real.

O mesmo `main.py` e os mesmos endpoints funcionam nos dois modos — só troca
a implementação do backend carregada no `lifespan`.

Para expor a API rodando dentro do Kaggle publicamente (para testar de fora
do notebook), o projeto inclui um notebook (`notebooks/ace_step_api_kaggle.ipynb`)
que sobe `uvicorn` em background e abre um túnel `ngrok`.

Limitações conhecidas e aceitas do Kaggle: sessão expira (~9-12h), e a URL
do túnel ngrok muda a cada nova sessão (plano free) — não há requisito de
persistência entre sessões neste projeto de teste.

## 3. Escopo funcional (o que a API cobre)

O ACE-Step tem vários recursos além de "texto → música". Foram investigados
na documentação oficial do projeto para decidir o escopo:

- **Tonalidade e BPM não são parâmetros estruturados do modelo.** Eles são
  descritos em texto livre dentro do campo de tags/prompt (ex:
  `"pop, 120 bpm, C major, acoustic guitar"`), como influência, sem garantia
  rígida de que o modelo obedeça exatamente. Não existe campo separado
  `key=` ou `bpm=` na API do modelo.
- **Não há separação de stems (voz/baixo/bateria/guitarra/violino) hoje.**
  O ACE-Step gera um único arquivo de áudio já mixado. Existe um recurso
  planejado chamado `StemGen` no roadmap oficial do projeto, mas ainda não
  foi lançado. Separação de stems real exigiria uma ferramenta externa
  (ex: Demucs) rodando como pós-processamento — fora de escopo deste
  projeto.
- **Voice Cloning foi deixado de fora do v1.** A documentação não deixa claro
  se é feito por inferência simples (áudio de referência) ou exige treinar
  uma LoRA específica por voz (processo de treinamento, não de inferência —
  o padrão observado em outras LoRAs do projeto, como `RapMachine`, é
  treino). Treino de LoRA é uma tarefa de outra natureza (horas de GPU,
  pipeline de treino) e não cabe no escopo de "API de teste de inferência".
  Fica documentado aqui como possível trabalho futuro, a ser investigado
  antes de decidir se é viável como endpoint de inferência.

**Confirmado como inferência pura (rápida, sem treino)** e incluído no v1,
um endpoint por operação, todos síncronos (a requisição bloqueia até o
áudio ficar pronto) e retornando `audio/wav` via `StreamingResponse`:

| Endpoint | Tipo de request | Campos principais |
|---|---|---|
| `POST /generate/text2music` | JSON | `tags`, `lyrics`, `duration`, `seed`, `steps`, `guidance_scale` |
| `POST /generate/retake` | JSON | igual a `text2music` + `variance` |
| `POST /generate/repaint` | multipart (arquivo de áudio + campos) | `audio_file`, `start_time`, `end_time`, `tags`, `lyrics` |
| `POST /generate/edit` | multipart | `audio_file`, `tags`, `lyrics`, `mode` (`only_lyrics` \| `remix`) |
| `POST /generate/extend` | multipart | `audio_file`, `left_extend_seconds`, `right_extend_seconds`, `tags`, `lyrics` (opcionais) |
| `POST /generate/audio2audio` | multipart | `audio_file`, `tags`, `lyrics` |

`lyrics` usa os marcadores de estrutura do próprio modelo: `[verse]`,
`[chorus]`, `[bridge]` (pode ser vazio para instrumental).

**Simplificação deliberada sobre `duration`:** o ACE-Step aceita
`duration=-1` para "duração aleatória". A API deste projeto não repassa
essa opção — `duration` é sempre um valor explícito e positivo (validado
por Pydantic). Isso mantém os testes determinísticos (dá para afirmar a
duração exata do áudio retornado). Se a duração aleatória for desejada no
futuro, é responsabilidade do chamador sortear um valor antes de enviar.

## 4. Estrutura do projeto

```
music_app/
├── CLAUDE.md                    # visão geral, stack, os 2 modos, comandos
├── README.md                    # como rodar local (mock) e no Kaggle (real)
├── pyproject.toml                # deps via uv; grupo opcional "kaggle" com acestep
├── .env.example                  # MUSIC_BACKEND=mock|acestep, PORT, ACE_STEP_CHECKPOINT_PATH
├── .gitignore
├── docs/superpowers/specs/       # este documento
├── app/
│   ├── main.py                    # FastAPI app, lifespan, middleware de log, rotas
│   ├── settings.py                # pydantic-settings
│   ├── schemas.py                 # Pydantic request/response por endpoint
│   ├── backends/
│   │   ├── base.py                 # Protocol com os 6 métodos
│   │   ├── mock_backend.py         # gera áudio sintético (sem GPU)
│   │   └── acestep_backend.py      # integração real com o pacote `acestep`
│   └── utils.py                   # ex: audio_array_to_wav_buffer, ler upload como array
├── tests/
│   └── test_api.py                # testa os 6 endpoints com MUSIC_BACKEND=mock
└── notebooks/
    └── ace_step_api_kaggle.ipynb   # sobe a API real no Kaggle + túnel ngrok
```

## 5. Backend abstraction

`app/backends/base.py` define um Protocol (ou ABC) com um método por
operação — a mesma assinatura para `mock_backend.py` e `acestep_backend.py`:

```python
class MusicBackend(Protocol):
    def text2music(self, tags: str, lyrics: str, duration: float,
                    seed: int | None, steps: int, guidance_scale: float
                    ) -> tuple[np.ndarray, int]: ...

    def retake(self, tags: str, lyrics: str, duration: float,
               seed: int | None, steps: int, guidance_scale: float,
               variance: float) -> tuple[np.ndarray, int]: ...

    def repaint(self, audio: np.ndarray, sample_rate: int,
                start_time: float, end_time: float,
                tags: str, lyrics: str) -> tuple[np.ndarray, int]: ...

    def edit(self, audio: np.ndarray, sample_rate: int,
             tags: str, lyrics: str,
             mode: Literal["only_lyrics", "remix"]) -> tuple[np.ndarray, int]: ...

    def extend(self, audio: np.ndarray, sample_rate: int,
               left_extend_seconds: float, right_extend_seconds: float,
               tags: str, lyrics: str) -> tuple[np.ndarray, int]: ...

    def audio2audio(self, audio: np.ndarray, sample_rate: int,
                     tags: str, lyrics: str) -> tuple[np.ndarray, int]: ...
```

- `main.py` escolhe a implementação no `lifespan`, olhando
  `settings.music_backend` (`"mock"` ou `"acestep"`), e guarda a instância
  única em `app.state`.
- `mock_backend.py`: para operações com áudio de entrada (repaint/edit/
  extend/audio2audio), usa a duração do áudio recebido (mais a extensão
  pedida, no caso de `extend`) para gerar um tom sintético de saída — não
  precisa "entender" o áudio de entrada, só validar o contrato de tamanho
  e formato da resposta.
- `acestep_backend.py`: assinatura exata dos métodos do pacote `acestep`
  (`ACEStep(...)`, `.generate(...)` etc.) será conferida contra a versão
  instalada durante a implementação — a doc pública descreve os
  parâmetros conceituais (tags, lyrics, duration, steps, guidance_scale,
  seed) mas não uma referência de API 100% fechada; a implementação deve
  validar contra `pip show acestep` / código-fonte instalado no Kaggle.

## 6. Tratamento de erros

- Parâmetros inválidos (ex: `duration <= 0`, `start_time > end_time`,
  `mode` fora de `only_lyrics`/`remix`) → `422` via validação Pydantic.
- Arquivo de áudio que não decodifica → `400` com mensagem clara.
- Falha do backend `acestep` (ex: OOM de VRAM) → `500` com o erro original
  logado, sem derrubar o processo.
- Middleware de log (adaptado do repo de referência) grava em
  `usage.csv`: request id, datetime, endpoint, tempo de resposta, status.

## 7. Testes

- `tests/test_api.py` usa `TestClient` do FastAPI com
  `MUSIC_BACKEND=mock` forçado via variável de ambiente/override de
  settings — roda em qualquer máquina, sem GPU e sem o pacote `acestep`
  instalado.
- Cobre os 6 endpoints: status 200, `content-type` correto
  (`audio/wav`), e que o áudio retornado tem a duração esperada (lendo o
  buffer com `soundfile`).
- Casos de erro (ex: `duration` negativo) cobertos com status 422.
- Não há teste automatizado contra o `acestep` real (exigiria GPU) — a
  validação desse caminho é manual, rodando o notebook no Kaggle.

## 8. Dependências

- `pyproject.toml` gerenciado com `uv`.
- Dependências sempre instaladas (ambiente local/mock): `fastapi`,
  `uvicorn`, `pydantic-settings`, `numpy`, `soundfile`, `pytest`,
  `httpx` (para `TestClient`).
- Dependência opcional (grupo `kaggle`), só instalada dentro do notebook:
  `acestep @ git+https://github.com/ace-step/ACE-Step.git`.

## 9. Notebook Kaggle

`notebooks/ace_step_api_kaggle.ipynb` contém células para:

1. Instalar `uv` e as dependências do grupo `kaggle` (inclui `acestep`).
2. Copiar/clonar o código de `app/` para o ambiente do notebook.
3. Definir `MUSIC_BACKEND=acestep` e o caminho do checkpoint.
4. Subir `uvicorn app.main:app --host 0.0.0.0 --port 8000` em background.
5. Abrir um túnel `ngrok` e imprimir a URL pública para testar de fora.

## 10. Fora de escopo (YAGNI / trabalho futuro)

- Separação de stems (StemGen ainda não lançado pelo ACE-Step).
- Voice Cloning (ambíguo entre inferência e treino de LoRA — investigar
  depois).
- Treinamento de LoRA customizada (RapMachine, Text2Samples, etc. — usar
  apenas se já vierem prontas, nunca treinar no v1).
- Persistência de jobs/fila assíncrona — todos os endpoints são síncronos.
- Autenticação/autorização na API (projeto de teste, uso local/pessoal).
- Deploy fora do Kaggle (ex: servidor próprio com GPU) — fora de escopo
  por agora.
