# Deploying the webapp (getting a public link)

The demo (`webapp/`) is a **FastAPI + PyTorch** server, not a static site — so
**Netlify / GitHub Pages won't run it**. Host it on **Hugging Face Spaces**:
free, no credit card, a permanent public URL, and it runs PyTorch fine. The
included `Dockerfile` is CPU-only and self-contained (weights are 16 MB + 9 MB,
no runtime downloads), so it also works on Render, Fly.io, or Cloud Run.

## Hugging Face Spaces — step by step (~5 minutes)

You need a free account at https://huggingface.co (sign up once).

1. **Create the Space.** https://huggingface.co/new-space → pick **Docker**,
   template **Blank**, visibility **Public**. Name it e.g. `melanoma-demo`.
   This creates a git repo with a `README.md` (it holds the `sdk: docker`
   frontmatter — **don't delete or overwrite that README**).

2. **Clone the Space and enter it:**
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/melanoma-demo
   cd melanoma-demo
   ```

3. **Copy the app into it** (from this repo, `G:\srip`). Copy these — and keep
   the Space's own `README.md`:
   - `Dockerfile`
   - `webapp/`  (this includes the model weights under `webapp/models/`)
   - `model/`
   - `preprocessing/`

4. **Track the weights with Git LFS** (HF wants LFS for binaries >10 MB):
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes
   ```

5. **Commit and push:**
   ```bash
   git add -A
   git commit -m "Melanoma classifier demo"
   git push
   ```

6. HF builds the container (~3–6 min the first time — watch the **Logs** tab).
   When it says *Running*, your link is:
   **`https://huggingface.co/spaces/<your-username>/melanoma-demo`**

That URL is the one to paste into the synopsis and weekly report.

## Notes

- The weights are your own trained model, so a public Space is fine. Want them
  private? Make the Space **private** — the link then only works while logged in
  as you, which is usually *not* what you want for a report reviewer, so prefer
  public.
- Free Spaces sleep after ~48 h idle and wake on the next visit (a few seconds).
  Fine for a demo link.
- **Other hosts:** the same `Dockerfile` deploys to Render (New → Web Service →
  Docker) or Fly.io (`fly launch`) — both have free/low tiers. HF Spaces is the
  least-friction for an ML demo.
