# The Melanoma Classifier Project — Explained from Scratch

*A complete walkthrough for someone who has never done machine learning. Every
number in this document comes from the project's own result files. Nothing here
is invented or estimated.*

---

## Before section 0: where this actually started

Before any of the code below existed, this project started as reading, not
coding. The `research papers/` folder (still in the repo) holds that first
step: 11 published papers on melanoma detection and dermoscopy image
preprocessing, collected first —

```
1-s2.0-S209012322500654X-main.pdf
372776.pdf
applsci-16-01819-with-cover.pdf
diagnostics-12-00344-v4.pdf
fmed-11-1495576.pdf
jimaging-11-00107.pdf
peerj-cs-1953.pdf
s41598-025-09938-4.pdf
s41598-025-91446-6.pdf
s41598-026-44545-x.pdf
syedmunazirshajahan.pdf
```

— followed by a literature-survey PPT built from them, in
`research papers/ppt/`: *"Literature Survey on Image Preprocessing Techniques
for Melanoma Detection."* That order — papers first, survey second, code
third — is the actual sequence: read what others had already tried, put the
survey together, and only then start building. It's also *why* preprocessing
in this project is not a single fixed step but a set of named, swappable
modes (`on`, `off`, `color_norm`, `devig`, `devig_ruler` — see §2.3): that
optionality traces back to the survey weighing multiple preprocessing
techniques against each other rather than picking one blind.

Everything from here on (§0 onward) is the part that was actually built.

---

## 0. The one-paragraph version

We built a computer program that looks at a photo of a skin spot and predicts
whether it is **melanoma** (a dangerous skin cancer) or something harmless. It
gets it right about as often as a good program of its type can — but it hit a
wall at a certain level of accuracy and stopped improving. The rest of this
project was a series of honest experiments to find out **why** it stopped, and
what (if anything) would push it further. Four experiments later, the answer is:
the wall is not caused by the obvious suspects. It is caused by the problem
itself being genuinely hard, plus a subtle bad habit the model picked up. This
document explains all of that, step by step, defining every term the first time
it appears.

---

## 1. The vocabulary you need first

Before any story, a few words. Each is defined again in context later, and all
of them are collected in the glossary at the end.

- **Melanoma** — a serious, potentially deadly form of skin cancer. In this
  project it is the "positive" thing we are trying to catch.
- **Benign / malignant** — benign means harmless; malignant means cancerous
  (dangerous). Melanoma is malignant. Most skin spots are benign.
- **Dermoscopy image** — a close-up photo of a skin spot taken through a special
  magnifying lens doctors use. Think of it as a very zoomed-in, well-lit photo of
  a single mole or lesion.
- **Lesion** — the medical word for a single skin spot (a mole, a mark, a
  growth). One lesion can be photographed more than once.
- **HAM10000** — the name of the public collection of dermoscopy images this
  project learned from. It contains **10,015 images**. Of those, about **11%**
  are melanoma. "HAM" stands for "Human Against Machine"; "10000" is roughly how
  many images it holds.
- **Model** — the program that learns to make the prediction. When we say "the
  model," we mean the melanoma classifier.

---

## 2. The starting point: the baseline model

### 2.1 What "the model" actually is

The model is a **neural network** — a large mathematical function with millions
of internal numbers ("weights") that get tuned until the function does something
useful. You don't program the rules by hand; you show it thousands of examples
and it adjusts its own weights to fit them. This adjusting-from-examples is
called **training**.

The specific network here is called **EfficientNet-B0**. It is a well-known,
off-the-shelf design for looking at images. "B0" is the smallest member of the
EfficientNet family (they go B0, B1, B2, B3, and up — bigger numbers mean bigger,
heavier networks). The images were fed in at **224×224 pixels** — a standard
small size that keeps training fast.

### 2.2 It didn't learn from a blank slate — transfer learning

We did **not** teach EfficientNet-B0 to see from nothing. We started from a copy
that had already been trained on **ImageNet** — a giant general collection of
millions of everyday photos (cats, cars, mushrooms, etc.). A network trained on
ImageNet already knows generic visual building blocks: edges, textures, color
blobs, shapes. Reusing that pre-learned visual knowledge for a new task is called
**transfer learning**, and a network that starts from ImageNet knowledge is
**ImageNet-pretrained**.

The analogy: instead of hiring someone who has never seen a photograph and
teaching them vision from birth, you hire someone who already has excellent
general eyesight and just teach them the one new specialty — "here is what
melanoma looks like."

### 2.3 How the training was staged: backbone, freezing, fine-tuning

A network like this has two parts:

- The **backbone** — the big stack of layers that does general "seeing" (the part
  inherited from ImageNet).
- The **head** — a small final part that turns "what the backbone saw" into the
  actual yes/no melanoma answer.

Training happened in two stages:

1. **Stage 1 (5 epochs):** the backbone was **frozen** — its inherited weights
   were locked and not allowed to change — while only the new head learned. An
   **epoch** is one complete pass through all the training images. Freezing the
   backbone first lets the fresh, untrained head catch up without disturbing the
   valuable ImageNet knowledge.
2. **Stage 2 (10 epochs):** the backbone was **unfrozen** (all of it) and allowed
   to adjust too. Letting the pretrained weights change to fit the new task is
   called **fine-tuning**. So the model trained for **15 epochs total** (5 + 10).

Two more knobs worth naming:

- **Learning rate** — how big a step the model takes each time it corrects
  itself. Too big and it overshoots; too small and it crawls. Stage 2 used a
  small, careful learning rate (0.0001) precisely because it was adjusting
  delicate pretrained weights.
- **"off" preprocessing** — "preprocessing" means any cleanup done to an image
  before the model sees it. "off" means we did the minimum: just resize the
  image, nothing else. (Later experiments tried turning some cleanup *on*.)

### 2.4 How well did it do? The baseline numbers

Here are the headline results for the baseline model. Each is explained right
after.

| Metric | Value |
|---|---|
| **AUC** | **0.9042 ± 0.0074** |
| Sensitivity | 0.9379 ± 0.0216 |
| Specificity at 95% sensitivity | 0.6619 ± 0.0395 |

**AUC** (Area Under the Curve) is the single most important score here. It is a
number between 0.5 and 1.0 that measures how well the model **separates** the two
groups — how good it is at giving melanomas higher "suspicion scores" than
benign spots.

> **How to read an AUC of 0.904:** pick one random melanoma and one random benign
> spot. AUC is the probability the model gives the melanoma the higher suspicion
> score. So 0.904 means it ranks them correctly about **90 times out of 100**.
> 0.5 would be a coin flip (useless); 1.0 would be perfect. 0.904 is a genuinely
> good score for this task. AUC is prized because it doesn't depend on where you
> draw the "call it cancer" line — it measures pure ranking ability.

**Sensitivity** is: of all the *actual* melanomas, what fraction did the model
catch? 0.9379 means it caught about **94%** of real melanomas. In cancer
screening this is the number you care about most — a missed melanoma is the
worst outcome.

**Specificity** is the mirror image: of all the *actual benign* spots, what
fraction did the model correctly wave through as harmless? Higher specificity
means fewer false alarms on healthy people.

There is a tension between the two: if you make the model quicker to shout
"cancer!", you catch more real melanomas (sensitivity up) but also raise more
false alarms (specificity down). So the project fixed a rule: **set the model to
catch 95% of melanomas, then measure specificity at that setting.** That is what
**"specificity at 95% sensitivity"** (written *spec@95*) means. The baseline's
0.6619 means: *when tuned to catch 95% of melanomas, it correctly clears about
66% of the benign spots.* The point where you draw the "cancer vs. not" line is
called the **decision threshold**.

> **How to read "± 0.007":** the ± number is the **standard deviation** — a
> measure of how much the result wobbled from run to run (here, across the five
> folds explained in Section 4). A small ± (like 0.007 on the AUC) means the
> result was stable and trustworthy. A big ± would mean "this number is shaky,
> don't lean on it." The ± is as important as the number itself: it tells you how
> seriously to take small differences later. **Roughly: a change smaller than the
> ± is probably just noise.** Hold onto that idea — it decides the verdict of
> every experiment below.

---

## 3. The problem that started everything

The baseline worked well. But a diagnostic tool called **Grad-CAM** revealed
something troubling.

**Grad-CAM** (think of it as "attention" or a heat-map) shows *which part of the
image the model actually looked at* when it made its decision. It paints the
image with a heat-map: bright where the model paid attention, dark where it
didn't. For a trustworthy melanoma model, the bright region should sit **on the
lesion** — the skin spot itself.

Instead, on several images the model's attention was hugging the **borders and
corners** of the photo, not the lesion in the middle. That is a red flag. It
suggests the model might be cheating — making its call based on something in the
background or the framing rather than the actual mole.

Why would it do that? Two suspects:

- **Artifacts** — things in the photo that aren't skin: a **ruler** (doctors
  sometimes lay a measuring ruler beside a lesion), ink marks, or a **vignette**
  (the dark circular shadow around the edge of some dermoscopy photos, from the
  lens). If melanoma photos happen to contain rulers more often than benign ones,
  a lazy model could "detect ruler → guess melanoma" and score well for the wrong
  reason. This kind of accidental, cheat-able correlation is called **data
  leakage** or a **shortcut**.
- **Positional bias** — even with no physical artifact, the model may have learned
  that "lesion pushed toward the edge of the frame" tends to mean melanoma, and
  started keying off *composition* instead of the lesion's actual appearance.

The whole rest of the project grew out of one question: **is the 0.904 ceiling
propped up by cheating, and if we stop the cheating, does the model get better,
worse, or stay the same?**

---

## 4. The measurement setup: why "honest" evaluation was built first

Before running any experiment, the project built a careful **evaluation
harness** — the machinery that scores the model. This matters more than it
sounds. A sloppy scoring setup can make a model look better than it is, and then
every experiment you run on top is measuring a lie. Here is each protective piece
and what it guards against.

### 4.1 Cross-validation and folds

You must never test a model on the same images you trained it on — it could just
memorize them and look brilliant while being useless on new patients. The clean
way to measure is **cross-validation**: split the data into chunks, train on some
chunks, test on a held-back chunk the model never saw.

This project used **5-fold** cross-validation. The data is divided into 5 parts.
The whole process runs 5 times ("5 folds"); each time, a different part is held
back for testing and the other parts are used for training. You end up with 5
independent-ish scores, and you report their average and their ± spread. That is
where every "± 0.007" in this project comes from: **it is the wobble across the 5
folds.**

### 4.2 The leak-proof way of splitting: lesion-aware splits

There is a trap. The same lesion can appear in **multiple photos** in HAM10000.
If one photo of a mole lands in the training set and another photo of the *same*
mole lands in the test set, the model has effectively seen the answer — that is
**data leakage** again, and it inflates the score.

The fix: split by **lesion ID**, not by image. Every photo of a given lesion is
forced into the *same* chunk. This is done with a method called
**StratifiedGroupKFold**:

- **Group** = keep all images of one lesion together (the anti-leakage part).
- **Stratified** = keep the melanoma percentage roughly equal (~11%) in every
  chunk, so no fold is accidentally starved of melanomas.

This "no lesion appears on both sides" discipline is what the project calls
**lesion-ID-aware splits**, and it is the single most important honesty
safeguard.

### 4.3 Three-way split inside each fold

Each fold actually carves its data into **three** parts, not two:

- **inner_train (~64%)** — used to train the model.
- **inner_val (~16%)** — a "calibration" set, used to tune two dials (below)
  *without* touching the final test data.
- **held_out (~20%)** — the true exam. Scored once, at the end. Never used for
  training or tuning.

Keeping calibration (inner_val) separate from the final exam (held_out) means the
dials get tuned honestly, on data that isn't the test.

### 4.4 The two dials: calibration and threshold

- **Calibration / temperature scaling** — a trained network's confidence numbers
  are often miscalibrated (it says "90% sure" when it's really 70% sure).
  **Temperature scaling** is a gentle one-number adjustment that fixes the
  confidence scale without changing the model's rankings. (Because it doesn't
  change rankings, it does **not** change AUC — a fact the project relied on.)
- **Decision threshold** — the cutoff for calling something melanoma, frozen on
  inner_val at the point that catches 95% of melanomas, then applied unchanged to
  the held_out exam. Frozen in advance so we can't cheat by picking the
  flattering cutoff after seeing the answers.

### 4.5 TTA — squeezing a bit more signal at test time

**TTA** (Test-Time Augmentation) means: at exam time, show the model a few
variations of each image (e.g. the original plus mirror-flips), get its opinion
on each, and average them. Averaging several looks is a bit more reliable than a
single look, and it slightly changes the AUC (unlike temperature scaling, TTA
*does* affect AUC because it changes the rankings). The project applied TTA
consistently so all arms are compared on equal footing.

### 4.6 The drift guard

Finally, an automatic sanity check the reports call a **drift guard**. Because the
5 folds share overlapping training data, one fold occasionally behaves oddly for
reasons unrelated to the experiment. The guard flags any fold whose behavior
drifted far from the others, so a single weird fold can be reported honestly
instead of silently skewing the average.

**Why all this matters:** every experiment below is a *comparison* — new model
vs. baseline. The comparison is only trustworthy because both sides run through
this identical, leak-proof, pre-frozen harness. The honesty is baked into the
ruler, not into the hope.

---

## 5. The four experiments

A note on how to read every verdict. The baseline AUC wobbles by about ±0.007
across folds. So a change in AUC **smaller than roughly 0.01 is inside the
noise** — it "reads neutral," meaning we can't tell it apart from random fold-to-
fold wobble. A result that shows no real effect is called a **null result** (or
"neutral"). A null result is not a failure — it is real, useful information: it
rules a suspect out. Every experiment here reports honestly, including the nulls.

One recurring caveat, stated up front: with only **5 folds** that *share*
training data, our statistical power is limited. A null here means "we couldn't
detect an effect at this scale," **not** "we proved there is zero effect." The
reports call this a **power ceiling**. It's the honest limit of a small study.

The symbol **Δ (delta)** just means "the change" — new value minus baseline
value. A negative Δ means the new thing scored lower.

---

### Experiment 1 — Remove the artifacts (masking)

**Question:** Is the model cheating off rulers and vignettes? If we hide those
artifacts, does its score change?

**What we did:** Built a detector that finds ruler marks and the circular
vignette, and blacked them out ("masked" them) before the model saw the image
(this cleanup mode is called `devig_ruler`). Then re-ran the full honest 5-fold
comparison: normal images vs. masked images.

First, the detector confirmed the artifacts are *real and suspicious*: rulers
were found in **650 images (6.5%** of the dataset), and images with a ruler were
**3.6× more likely** to be melanoma than average. That is exactly the kind of
accidental correlation a model could exploit as a shortcut.

**Result:**

| Arm | AUC |
|---|---|
| Baseline (normal) | 0.9042 ± 0.0074 |
| Masked (artifacts hidden) | 0.9054 ± 0.0101 |

The change was **ΔAUC = +0.0012** — essentially zero, and far smaller than the
±0.007 fold noise.

**What it means:** Even though the artifacts are real and melanoma-enriched, the
model **was not leaning on them** to make its calls. Hiding them changed nothing.
This is an **honest negative result**: it clears the artifacts as the cause of
the ceiling.

But the Grad-CAM follow-up found the deeper problem. Of six hard missed cases,
**three had no physical artifact at all** — yet the model was still staring at the
image borders. That points to the second suspect from Section 3: a
**positional / compositional bias**. The model learned that lesions composed near
the frame edge lean melanoma, independent of the lesion's actual look. You can't
erase that with masking, because there's nothing physical to erase. That finding
set up Experiment 2.

---

### Experiment 2 — Force the model to look around (spatial augmentation)

**Question:** If we disrupt the model's fixation on framing and position, will its
attention move onto the lesion — and will accuracy improve?

**What we did:** Used stronger **data augmentation**. Augmentation means feeding
the model randomly altered copies of the training images — zoomed, cropped,
rotated — so it can't rely on any fixed framing and is forced to recognize the
lesion itself wherever it lands. This arm used aggressive random cropping (down to
55% of the image) and rotations up to 30°.

**Result:**

| Arm | AUC |
|---|---|
| Baseline | 0.9042 ± 0.0074 |
| Stronger augmentation | 0.8983 ± 0.0087 |

**ΔAUC = −0.0059** (± 0.0097). Negative, but well inside the fold noise — a
**neutral / null result** on accuracy.

**What it means, in two parts:**

1. On the *score*, stronger augmentation didn't help (and the tiny dip is within
   noise). So it's not a free lever for beating the ceiling.
2. But on *behavior*, it did what we hoped: Grad-CAM showed the model's attention
   **moved onto the lesions**. On the hard missed cases the "center-energy" (how
   much attention sat in the middle, on the lesion, vs. the edges) rose by
   **+0.097 on average**, and **2 of 6** previously-missed melanomas flipped to
   correct.

So augmentation fixed the *bad habit* (edge-staring) without moving the *bottom-
line number*. That is an important clue: it means the edge-staring habit was not
actually what was holding the score down. The ceiling lives somewhere else.

---

### Experiment 3 — Use a bigger model (B3 backbone)

**Question:** Is the model simply too small? Would a bigger, higher-capacity
network learn more and break through the ceiling?

**What we did:** Swapped EfficientNet-B0 for **EfficientNet-B3** — a substantially
larger network with more layers and more internal weights (more **capacity**,
i.e. more room to learn complex patterns) — keeping everything else the same, at
the same 224-pixel image size.

**Result:**

| Arm | AUC |
|---|---|
| Baseline (B0) | 0.9042 ± 0.0074 |
| Bigger model (B3) | 0.8956 ± 0.0132 |

**ΔAUC = −0.0086** (± 0.0155) — again inside the noise band, a **null result**.
The bigger model was not better.

There was also a telling side-finding. An **overfitting** check flagged **all 5
folds**. **Overfitting** is when a model starts *memorizing* the training images
instead of learning general patterns — it aces the material it has seen but
doesn't get better on new patients. The check compares performance on training
data vs. held-back data; a widening gap means memorization. The bigger B3 model,
with all its extra capacity, mostly used that capacity to memorize.

**What it means:** Throwing a bigger model at the problem doesn't help — with this
much data, the extra capacity just overfits. Model size is not the binding
constraint. (Grad-CAM was mildly encouraging: 3 of 6 hard cases flipped correct,
attention center-energy +0.056 — but the headline score didn't move.)

---

### Experiment 4 — Is it a data-quantity problem? (the learning curve)

By now three suspects were cleared: artifacts, framing bias, model size. One big
suspect remained — **maybe the model just needs more data.**

**Question:** Is the 0.904 ceiling a limit of *how much training data we have*? If
we had more, would the score keep climbing?

**What we did:** Built a **learning curve** — the standard way to answer exactly
this. A learning curve retrains the *same* model on increasing amounts of data
and plots accuracy against data size. We trained on **25%, 50%, 75%, and 100%** of
the available training data and measured AUC at each step. The subsets were
**nested** (the 25% set sits inside the 50% set, inside the 75%, inside the 100%),
kept the melanoma percentage fixed at ~11%, and respected the lesion-grouping
rule — so the only thing changing is *quantity*. To steady the noisy small-data
points, the 25% and 50% runs were each repeated with different **random seeds** (a
seed is the starting shuffle/initialization number; changing it gives an
independent run, so averaging several tells you the true level rather than one
lucky or unlucky draw).

The shape of that curve is the whole answer:

- If accuracy is **still climbing at 100%**, we're data-starved — more data would
  help.
- If accuracy has **flattened out** before 100% (hit an **asymptote** — a level it
  approaches and stops rising past, showing **diminishing returns** where each new
  batch of data buys less and less), then **quantity is not the problem.**

**Result:**

| Training data used | mean AUC | ± |
|---|---|---|
| 25% | 0.8627 | 0.0146 |
| 50% | 0.8862 | 0.0094 |
| 75% | 0.8979 | 0.0093 |
| 100% | 0.9042 | 0.0074 |

Look at the *steps between* the numbers — that's where the story is:

- 25% → 50%: **+0.0235** (a big jump)
- 50% → 75%: **+0.0117** (smaller)
- 75% → 100%: **+0.0063** (smaller still — and now *inside* the fold-noise band)

**What it means:** classic **diminishing returns**. Early data helped a lot; the
last quarter barely moved the needle. The final step (75%→100%, +0.0063) is
smaller than the pooled fold noise (±0.0102), so by the honest rule the curve has
gone **flat**. **Verdict: data quantity is NOT the binding constraint at this
scale.** Doubling the dataset would, by the trend of this curve, buy very little.

> **How to read the 100% row:** it wasn't retrained — it reuses the original
> baseline's exact per-fold scores as the anchor point (they are the same model on
> the same data). So the 100% point is guaranteed to equal the baseline; the real
> test is whether the 75% point sits just below it on a smoothly flattening curve.
> It does.

*(Power caveat, same as always: 5 shared folds, and the 25% points are the
noisiest because each fold then contains only a few dozen melanomas. A flat read
at N=5 is a power ceiling, not absolute proof — but it's the honest best estimate.)*

---

## 6. Putting it together: the thesis and its limits

Four suspects were lined up for the 0.904 ceiling. One by one, the honest harness
cleared them:

1. **Artifacts (rulers/vignettes)?** No — hiding them changed nothing (Δ +0.0012).
   The model wasn't cheating off them.
2. **Framing / positional bias?** Real, and augmentation fixed it — but fixing it
   didn't raise the score (Δ −0.0059). So it wasn't what capped accuracy.
3. **Model too small?** No — a bigger model didn't help and just overfit (Δ
   −0.0086).
4. **Not enough data?** No — the learning curve flattened before 100% (last step
   +0.0063, inside noise).

**The conclusion:** the ceiling around AUC 0.904 is **not** an artifact problem,
**not** a framing problem, **not** a capacity problem, and **not** a quantity
problem. By elimination, what remains is **data quality and the intrinsic
difficulty of the task** — some melanomas simply look, in a single dermoscopy
photo, genuinely ambiguous, and no amount of the four levers above resolves that.
Roughly 1 case in 10 sits in that hard zone, and that is where the wall is.

**The honest limits of that conclusion:**

- Every experiment is a **null result at N=5 folds** that share training data.
  That is a **power ceiling**: we're saying "we could not detect an effect,"
  which is weaker than "there is provably no effect." A larger, fully independent
  study could in principle surface a small effect we can't see.
- "Data quality / intrinsic difficulty" is reached by *elimination*, not by a
  direct measurement of difficulty. It's the best-supported explanation left
  standing, not a directly proven one.
- The class **imbalance** — only ~11% melanoma — genuinely makes the problem
  harder: the model sees roughly 8 benign spots for every 1 melanoma, so the
  interesting cases are the minority, and there are only a few dozen melanomas per
  fold at the small-data end. This is why the error bars fan out at 25%.

None of these caveats are weaknesses hidden in a footnote — they are stated in
every one of the project's own reports. Honesty about what a result *can't* say is
the point of the whole harness.

---

## 7. Where the project stands now

- A solid, honestly-measured **baseline** exists: EfficientNet-B0, AUC **0.9042 ±
  0.0074**, catching ~94% of melanomas.
- Four experiments are **done and reported**: artifact masking, spatial
  augmentation, bigger backbone (B3), and the learning curve. All four are honest
  nulls on the accuracy score — which together tell a consistent story about
  *why* the ceiling exists.
- One concrete behavioral **improvement** was found (augmentation moves attention
  onto the lesion), even though it didn't raise the headline number — worth
  keeping for trustworthiness reasons even if not for accuracy.
- **Open, not-yet-run follow-ups** noted in the reports: a regularized B3 (with
  weight-decay / early-stopping to curb the overfitting seen in Experiment 3), a
  native-resolution B3 run (to separate "capacity" from "image size"), and
  external validation on a *different* dataset (to test whether 0.904 holds up on
  images from another source). These are future work, deliberately not auto-run.

The overarching lesson is methodological as much as medical: a good result is not
just a high number, it's a number you've *tried hard to break and couldn't*. This
project spent four experiments trying to break its own ceiling honestly, and in
failing to break it, learned exactly what the ceiling is made of.

---

## 8. From model to demo: building the web app

Everything above is the research side — a model trained and honestly measured.
What follows is what happened next: turning that model into something a person
can actually point a browser at, upload a photo to, and get an honest reading
from. It lives in its own folder, `webapp/`, built after the four experiments
were done. This section, like the rest of the document, is compiled from the
project's own files (`webapp/README.md`, `app.py`, `inference.py`) — no number
here is invented either.

### 8.1 The brief, and the tool choice

The ask was a demo: upload an image, pick a model, toggle preprocessing,
compare. Two frameworks were weighed for building it — **Gradio** (a
ready-made ML-demo UI library — fast to stand up, but low control over layout
and wording) versus **FastAPI + Jinja2 + plain JavaScript** (a small Python web
server rendering real HTML/CSS — slower to build, full control). Because the
safety language (disclaimers, error rates, the off-distribution caveat below)
needed exact placement and exact wording, not whatever a template library
allows, the second option won. No new dataset, no new training — the web app's
only job is to *serve* the already-trained models faithfully.

The whole thing was built **isolated** from the research code: the two serving
checkpoints are *copies* dropped into `webapp/models/`, and `model_output/` —
the real training output — was never touched by any of this.

### 8.2 Not a second scoring system — the same one, reused

The riskiest way to build a demo like this would be to reimplement the scoring
math by hand and hope it matches. Instead, `webapp/inference.py` **imports and
reuses** the exact research pipeline: the same `build_model` constructor, the
same evaluation transform (resize → tensor → ImageNet normalization) and the
same 4-flip test-time-averaging and temperature-scaling formulas that
`finalize.py` and `model/dataset.py` already use. Nothing was rewritten from
scratch; the honest math was carried over, not re-derived.

Each model's **temperature** and **decision threshold** — the two dials
explained in §4.4 — are read straight from that model's own `deploy.json`,
never recomputed by the web app. Two models are registered, each fully
self-contained:

| Model | Temperature | Threshold | TTA |
|---|---|---|---|
| EfficientNet-B0 (primary) | 1.893 | 0.194 | on |
| MobileNetV2 (baseline) | 1.450 | 0.156 | off |

**EfficientNet-B3** — the bigger model from Experiment 3 — is deliberately
**left out** of the demo. Calibrating it needs the original HAM10000
validation split fitted the way §4.4 describes, which this demo doesn't have
access to. Rather than show it with a made-up threshold, the app's explainer
panel says plainly why it's absent. Leaving a known gap disclosed, instead of
quietly papering over it, is the same instinct as reporting the four null
results in §5 instead of hiding them.

A script called `parity_check.py` is the proof the reuse actually worked: it
re-runs the web app's own inference path over the same held-out test images
`finalize.py` scored, for every registered model, and checks the resulting
AUC / sensitivity / specificity against that model's `deploy.json` — expected
to match within 1e-3. That's the same discipline as the drift guard in §4.6,
aimed at a different failure mode: not "did a fold behave oddly," but "does the
serving code actually compute what the research code computed."

### 8.3 The Calibration Readout, and staying honest off-distribution

The interface's signature element is a **Calibration Readout**: a 0→1 bar
showing the calibrated probability as a marker, the model's own frozen
threshold as a labeled gate, and the naive 0.5 cutoff as a faint tick marked
"not used" — visually making the point from §2.4, that the decision line is
0.194 (or 0.156), never a generic 50%.

The web app also lets a visitor turn on the same real preprocessing steps used
elsewhere in this project — color normalization, vignette removal, ruler
masking, hair removal, denoising, contrast enhancement — via
`preprocessing_bridge.py`, which calls the *actual* functions in
`preprocessing/preprocess.py`, not a reimplementation. But both registered
models were trained and calibrated with preprocessing **off** (§2.3). So the
moment any toggle is switched on, the input the model sees is no longer the
kind of input it was calibrated for — what §4 would call **off-distribution**.
The app never hides this: it marks the reading **PROVISIONAL**, hatches the
readout, and adds an explicit caveat that the number is illustrative, not
backed by the reported AUC. It's the same rule as §4.4's frozen threshold,
applied to a new situation — don't show an authoritative-looking number where
the honesty guarantees no longer hold.

### 8.4 Comparing models, side by side

A later addition lets a visitor run **both** registered models on the same
image at once. It does not average or combine their answers into one score —
each model's probability, threshold, and stats stay entirely its own, exactly
as if requested separately. The only thing added is a single flag: whether the
two models' own threshold decisions **disagree**. Disagreement is disclosed,
not resolved by a fake combined verdict.

### 8.5 Where the demo stands now

The app was built in stages — a working skeleton first, then the designed
interface and safety copy, then the model selector, then the preprocessing
toggles, then the compare view — and later given a purely visual re-skin (an
editorial, instrument-panel look) with `parity_check.py` re-run afterward to
confirm the redesign changed nothing about what number gets shown. It runs
locally only (`uvicorn webapp.app:app`), reads two real calibrated models, and
carries every honesty rule from the research side into the interface: the
probability is never shown without its model's known error rates, the
threshold is never a naive 50%, and preprocessing that pushes the input
off-distribution is always disclosed as such.

---

## Appendix A — Glossary (every term, one line each)

| Term | Plain meaning |
|---|---|
| Melanoma | A dangerous, malignant skin cancer — the thing we're trying to detect. |
| Benign / malignant | Benign = harmless; malignant = cancerous/dangerous. |
| Dermoscopy image | A zoomed-in, well-lit close-up photo of a single skin spot. |
| Lesion | A single skin spot (mole/mark); can be photographed multiple times. |
| HAM10000 | The public set of 10,015 dermoscopy images used here (~11% melanoma). |
| Model / neural network | The trainable math function that makes the prediction. |
| Training | Adjusting the model's internal weights by showing it labeled examples. |
| EfficientNet-B0 / B3 | Off-the-shelf image-network designs; B0 is small, B3 is bigger. |
| Transfer learning | Reusing a model already trained on other images as a starting point. |
| ImageNet-pretrained | Started from knowledge learned on ImageNet, a huge general photo set. |
| Backbone | The big "seeing" part of the network (inherited from ImageNet). |
| Head | The small final part that outputs the melanoma yes/no answer. |
| Freezing / unfreezing | Locking weights so they can't change / releasing them to change. |
| Fine-tuning | Letting pretrained weights adjust to the new task. |
| Epoch | One full pass through all the training images. |
| Learning rate | How big a correction step the model takes each update. |
| Preprocessing ("off") | Image cleanup before the model sees it; "off" = just resize. |
| Random seed | The starting shuffle/init number; changing it gives an independent run. |
| AUC | 0.5–1.0 score for how well the model ranks melanomas above benign spots. |
| Sensitivity | Fraction of real melanomas the model catches. |
| Specificity | Fraction of real benign spots the model correctly clears. |
| Spec@95 | Specificity measured when the model is tuned to catch 95% of melanomas. |
| Decision threshold | The cutoff score for calling a spot "melanoma." |
| Class imbalance / prevalence | Melanoma is only ~11% of images — the rare, hard minority. |
| Cross-validation / 5-fold | Train/test on rotating chunks so nothing is tested on its own training data. |
| StratifiedGroupKFold | Splitting that keeps lesions grouped and melanoma % balanced. |
| Data leakage | When test info sneaks into training and inflates the score. |
| Lesion-ID-aware split | Forcing all photos of one lesion into the same chunk (anti-leakage). |
| Calibration / temperature scaling | A one-number fix to the model's confidence; doesn't change AUC. |
| TTA | Averaging the model's opinion over flipped/cropped copies at test time. |
| Drift guard | Auto-flag for a fold that behaved oddly vs. the others. |
| Grad-CAM / attention | Heat-map of which image region the model actually looked at. |
| Overfitting | Memorizing training images instead of learning general patterns. |
| Data augmentation | Feeding randomly altered image copies so the model can't rely on framing. |
| Artifact (ruler / vignette) | Non-skin stuff in the photo (a ruler, or the lens's dark circular edge). |
| Delta (Δ) | The change: new value minus baseline value. |
| Standard deviation (±) | How much a result wobbled across folds; a stability/trust measure. |
| Null / neutral result | No detectable effect — a real, useful finding, not a failure. |
| Power ceiling | The study is too small to detect small effects — "can't tell," not "isn't there." |
| Learning curve | Plot of accuracy vs. amount of training data. |
| Diminishing returns | Each extra batch of data helps less than the last. |
| Asymptote | A level a curve approaches and stops rising past. |
| deploy.json | The one file per model holding its frozen temperature, threshold, TTA setting, and stats — the single source of truth the web app reads from, never recomputes. |
| Calibration Readout | The web app's 0→1 bar showing the calibrated probability against the model's own frozen threshold (never a naive 0.5). |
| Parity check | Re-running the web app's inference path over the same held-out images the research code scored, to confirm the two match — a bug check for the serving code, not a new experiment. |
| Off-distribution (web app) | Input unlike what a model was trained/calibrated on (e.g. preprocessing a model that was calibrated on raw images) — flagged PROVISIONAL rather than shown as an authoritative number. |

## Appendix B — Experiment summary table

| Experiment | Question | Result (vs. baseline AUC 0.9042) | What it means |
|---|---|---|---|
| **1. Artifact masking** | Is the model cheating off rulers/vignettes? | Masked AUC 0.9054, **Δ +0.0012** (within noise) | Not cheating off artifacts — a clean null. But 3/6 hard misses had no artifact → a separate positional bias. |
| **2. Spatial augmentation** | Does breaking the framing habit help? | AUC 0.8983, **Δ −0.0059** (within noise) | No score gain, **but** attention moved onto the lesion (+0.097 center-energy, 2/6 misses fixed). Fixed the habit, not the ceiling. |
| **3. Bigger model (B3)** | Is the model too small? | AUC 0.8956, **Δ −0.0086** (within noise) | No gain; bigger model overfit on all 5 folds. Capacity isn't the limit. |
| **4. Learning curve** | Do we just need more data? | 25%→0.863, 50%→0.886, 75%→0.898, 100%→0.904; last step **+0.0063** (within noise) | Curve flattened → **quantity isn't the limit.** Diminishing returns already set in. |

**Overall:** four suspects tested, four cleared. The ~0.904 ceiling is best
explained by **data quality and the intrinsic difficulty** of telling some
melanomas apart in a single photo — stated honestly, within the power limits of a
5-fold study.

---
*EXPLAINER.md — a teaching summary compiled from the project's own reports and
result files (report.md, report_augment.md, report_backbone.md,
report_learning_curve.md, and the summary JSONs); for the opening section, from
the contents of research papers/ and research papers/ppt/; and — for §8 — from
webapp/README.md, webapp/app.py, and webapp/inference.py. No numbers were
invented; all trace back to those files. No code, model, or data was changed
to write this.*
