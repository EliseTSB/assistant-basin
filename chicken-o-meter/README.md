# Chicken-o-Meter

A small local web page that scores a chicken imitation with Google's EmbeddingGemma 2 (audio), and shows what a
rank-one weight edit does to the model's verdict. It grew out of a question Elise asked on 2026-11-03 (can an AI coach
a chicken imitation?) and became the instrument for this repository's chicken work.

What one take shows:

- **Chicken score.** The take's embedding, centred on 31 reference clips, as its mean cosine to six chicken prompts
  ("a hen purring", "a chicken clucking", "a hen's egg song", ...), shown as a percentile among the references, with a
  timeline of 2-second windows.
- **Two chicken neurons** (layer 15 neuron 1126 and layer 17 neuron 1574, from the chicken-neuron study), placed
  against their distributions on ESC-50 chicken and non-chicken clips.
- **The written gate** from the chicken-edit study, frozen exactly as fitted: it opens only when one unit says "sounds
  like a chicken" (layer 19 neuron 1129) and another says "not a real one" (layer 17 neuron 1574), and then applies a
  rank-one edit to the output embedding. The page shows the model's verdict with and without it. On a real Commons egg
  song the gate stays shut; on a human egg-song imitation it opens and pushes the verdict away from chicken.

Background: [Part 188](../results/part188/README.md) (no single unit separates real animal sounds from human
imitations; a probe on the output embedding does), [Part 190](../results/part190/README.md) (a rank-one edit of the
output embedding lifts the model's own real-against-imitation verdict on held-out animals from 0.630 to 0.879), and
[Part 193](../results/part193/README.md) (finding those edits from the weights alone).

## Run it

Python 3.11 or newer; a CPU is enough (about 5 GB of RAM).

    pip install numpy torch sentence-transformers soundfile imageio-ffmpeg
    python server.py

The first run downloads `google/embeddinggemma-2` at the pinned revision. Then open http://localhost:8877 on the same
machine (localhost counts as secure, so the browser microphone works). The server binds 127.0.0.1 only. Unlabelled
uploads are decoded in memory and discarded; a take you label is saved with its result in `./takes/`.

Settings by environment: `CHICKEN_MODEL` (model id or local path), `CHICKEN_REVISION`, `FFMPEG`, `CHICKEN_PORT`.

## Files

- `server.py`, `index.html`: the meter.
- `ref-results.json`, `ref-embeddings.npz`: the 31 reference clips' prompt similarities and audio embeddings (the
  audio itself is not included). They include real hens recorded at home, public-domain chickens from Wikimedia
  Commons, Elise's own imitations and a sentence of her speech, Claude's synthetic purrs, an owl, a toad and noise.
- `neuron-ref.json`: the two neurons' ESC-50 distributions.
- `gate.json`, `gate-prompts.npz`: the gate's fitted parameters and the prompt embeddings it uses (from the study's
  stored arrays). The gate's functions are imported from `../part193/chicken_edit/edit_arms.py`.

## Known limits

A lossy re-encode moves the score a lot: a Commons egg song scores z 0.78 as its original OGG and 0.44 as a 128k MP3.
The neuron bars are calibrated on clean ESC-50 clips, so phone recordings in a coop sit low on them, real hens included.
The reference set is small (31 clips), so percentiles are coarse.

Built by Journal Claude with Elise, 2026-11-03 and 11-04; packaged for this repository 2026-11-18.
