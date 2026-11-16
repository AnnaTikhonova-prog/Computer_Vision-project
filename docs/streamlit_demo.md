# Live breed demo

A small English-language app: upload a photo (or choose a bundled cat/dog),
click **Find the breed**, and see three scores from a real ResNet50 inference.
Two original pixel pets walk across the bottom. No account, dataset download,
API key, or GPU is needed to run it. The first prediction loads the model;
later predictions reuse it.

## Run locally

Use 64-bit **Python 3.13**, from the repository root. Install Git LFS first.
The demo has its own small dependency file; the training environment is unchanged.

```powershell
# If you do not have a checkout yet:
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/AnnaTikhonova-prog/Computer_Vision-project.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location Computer_Vision-project

git lfs install --local
git lfs pull --include="deliverables/c1/runs/lr_1e-4/seed_42/best.pt"
py -3.13 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r demo/requirements.txt
.\.venv-demo\Scripts\python.exe -m streamlit run demo/app.py
```

Open **http://localhost:8501**. Keep that terminal running. Subsequent offline
launches only require the last command, after dependencies and weights have
been downloaded. On Linux use `python3.13 -m venv .venv-demo` and
`.venv-demo/bin/python` instead of the Windows interpreter path.

## Deploy to Streamlit Community Cloud

1. Commit and push `demo/`, `.streamlit/config.toml`, and this documentation
   to GitHub. The existing checkpoint is already tracked by Git LFS; do not
   upload another model or the raw dataset.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/), connect the
   GitHub account with access to this repository, and choose **Create app**
   / **Deploy a public app from GitHub**.
3. Repository: `AnnaTikhonova-prog/Computer_Vision-project`.
   Branch: `main` (or the branch containing the demo).
   Main file path: **`demo/app.py`**.
4. In **Advanced settings**, choose **Python 3.13**. No secrets are required.
   Deploy and wait for installation and the Git LFS checkout to finish.
5. Open the app, choose **Cat**, and click **Find the breed**. Repeat with
   **Dog**, then upload a new image to verify live inference.

Community Cloud supports Git LFS. It finds `demo/requirements.txt` beside
the entrypoint before the root training requirements. The demo pins CPU
PyTorch wheels to avoid installing CUDA packages. The repository still
contains other LFS weights, so the initial cloud checkout may take time.
See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[dependency search rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
and [Git LFS support](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Presentation-day walkthrough (about 60 seconds)

1. Start the local app before presenting. Run one prediction to warm the model.
   Keep the local app available even when showing the cloud URL.
2. Say: **“This demo identifies ten cat and dog breeds from a photo.”**
3. Click **Find the breed** for the cat example; switch to the dog and repeat.
4. Upload an unseen JPG, PNG, or WebP from your device and run it live.
   Invite the TA to supply another image; the upload replaces the example.
5. Say: **“These are model scores, not calibrated certainty. It can confuse
   similar breeds and always chooses among its ten known classes.”**

This is an interactive demo, not a prerecorded result. Every button click
runs a forward pass on the currently displayed image. No retraining happens.
For a fully offline presentation, rehearse using the local launch command
with Wi-Fi disconnected. The pixel pets use only local SVG and CSS.

## Model and examples

- Fixed model: **C1, augmentation on, learning rate 1e-4, seed 42**;
  `deliverables/c1/runs/lr_1e-4/seed_42/best.pt` (about 94 MB).
  This is a single retained checkpoint, not the three-seed summary or an ensemble.
  No model/configuration was retrained or tuned for the demo.
- SHA-256: `d4cda19030d5ee60eb2c8b7ec79a3444cf366f5922d1a9f09acdac2a61ec1384`.
  The app verifies the hash and class mapping before loading.
- Preprocessing is torchvision's ImageNet V2 transform, matching evaluation:
  resize shorter side to 232, center crop 224, RGB tensor, ImageNet normalization.
  Phone EXIF orientation is applied first. The original photo is displayed;
  inference uses the center crop, so keep the pet near the center.
- Bundled examples are unchanged `Birman_1.jpg` and `pug_10.jpg`, from the
  **training** manifest. They demonstrate interaction, not independent accuracy.
  New uploads are processed in memory; the app does not write them to disk or
  store them in a shared prediction cache. The model itself is shared read-only.
- Supported classes: British Shorthair, Russian Blue, Siamese, Birman,
  American Bulldog, Pug, Sphynx, Maine Coon, Beagle, Samoyed. There is no
  unknown-class detector. A non-pet image also receives a breed prediction.
- Oxford-IIIT Pet / Parkhi et al., *Cats and Dogs*, CVPR 2012;
  [dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/),
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  See [third-party notices](../THIRD_PARTY_NOTICES.md). Pixel sprites are original code.

## Troubleshooting

- **Missing/incomplete model:** run the exact `git lfs pull --include=...`
  command above. A small text pointer is not the checkpoint. On Cloud, inspect
  the build logs for Git LFS errors and check repository/LFS access.
- **Wrong packages or CUDA downloads:** use `demo/app.py` as the entrypoint,
  Python 3.13, and `demo/requirements.txt`; do not use the root training file.
- **First request is slow:** model loading is lazy. Warm it once before the talk.
  Cloud apps may need to wake after inactivity; keep the local copy ready.
- **Image rejected:** upload a valid JPG/PNG/WebP under 10 MB and 20 megapixels.
  A new upload takes priority; clear it with the uploader's X to use examples.
- **Unexpected breed:** use one centered pet from the supported classes.
  Scores can be confidently wrong; they do not validate an animal's pedigree.

## Check before presenting

```powershell
.\.venv-demo\Scripts\python.exe -m unittest discover -s tests -p test_demo.py -v
```

The tests exercise both examples through the actual checkpoint, compare the
serving path with a frozen evaluator prediction and shared preprocessing, check
invalid image handling, and click the app's controls with Streamlit AppTest.
They do not alter any frozen reports, configurations, or weights.

Verified locally on 16 November 2026 with Python 3.13.15, Streamlit 1.66.0,
PyTorch 2.11.0+cpu and torchvision 0.26.0+cpu: all six demo tests passed,
`pip check` passed, and a headless Chrome session checked both examples,
file upload, invalid-file feedback, moving sprites, and a 390-pixel viewport
without horizontal overflow or JavaScript errors. Community Cloud deployment
itself has not been performed; follow the steps above in your account.
