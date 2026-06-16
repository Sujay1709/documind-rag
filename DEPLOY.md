# Deploying DocuMind to Hugging Face Spaces (free, Docker)

This deploys the whole app — Streamlit UI **and** a bundled Ollama server — into a
single Docker Space, so it's reachable from a public URL anywhere.

> **Heads-up on the free tier.** Spaces' free hardware is CPU-only (2 vCPU, 16 GB
> RAM). `llama3.2:3b` runs there but answers are slow, and unless you add
> **persistent storage**, the models re-download on every cold start (a few
> minutes). For a snappier demo, upgrade the Space hardware or attach a persistent
> disk. For production speed you'd point `DOCUMIND_OLLAMA_BASE_URL` at a GPU-backed
> Ollama host instead.

## What's in `deploy/hf-spaces/`

| File | Purpose |
|------|---------|
| `Dockerfile` | Space image: Python + Ollama + the app, listening on port 7860. |
| `start.sh` | Starts Ollama, pulls the models, then launches Streamlit. |
| `README_SPACE.md` | The Space's `README.md` (contains the required HF metadata header). |

## Steps

1. **Create the Space**
   - Go to https://huggingface.co/new-space
   - Owner: your account · Space name: `documind` · License: MIT
   - **SDK: Docker** → *Blank* template · Hardware: *CPU basic (free)* → **Create Space**

2. **Get the project into the Space repo.** A Space is a git repo. The simplest way
   is to push this project into it with the Spaces Dockerfile at the repo root:

   ```bash
   # clone your new (empty) Space
   git clone https://huggingface.co/spaces/<your-hf-username>/documind
   cd documind

   # copy the app from your GitHub checkout (adjust the path)
   cp -r ~/rag-app/* ~/rag-app/.streamlit ~/rag-app/.gitignore .

   # HF needs the Dockerfile and README at the repo ROOT:
   cp deploy/hf-spaces/Dockerfile ./Dockerfile
   cp deploy/hf-spaces/README_SPACE.md ./README.md

   git add -A
   git commit -m "Deploy DocuMind to HF Spaces"
   git push
   ```

   (You'll authenticate the push with a Hugging Face access token — create one at
   https://huggingface.co/settings/tokens with *write* scope, and use it as the
   git password.)

3. **Watch it build.** Open the Space → **Logs**. You'll see Ollama start, the
   models download, then Streamlit launch. When the build finishes, the app loads
   in the Space's **App** tab.

## Optional: faster / persistent

- **Persistent models:** Space → Settings → add **Persistent Storage**. The image
  already writes models to `/data/ollama`, so they'll survive restarts.
- **Bigger/faster hardware:** Space → Settings → Hardware → pick a paid tier.
- **Custom models:** set Space *Variables* `DOCUMIND_CHAT_MODEL` /
  `DOCUMIND_EMBEDDING_MODEL` to change what `start.sh` pulls.

## Local check before deploying

You can build and run the Space image locally to verify it:

```bash
docker build -f deploy/hf-spaces/Dockerfile -t documind-space .
docker run --rm -p 7860:7860 -v documind_data:/data documind-space
# open http://localhost:7860  (first run downloads models)
```
