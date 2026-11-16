"""Run from the repository root: python -m streamlit run demo/app.py."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.inference import load_predictor, predict, read_image

ASSETS = Path(__file__).resolve().parent
EXAMPLES = {"Cat": "Birman_1.jpg", "Dog": "pug_10.jpg"}

st.set_page_config(page_title="Pet / A little breed finder", page_icon="🐾", layout="centered")
st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def predictor():
    return load_predictor()


st.markdown('<div class="eyebrow">OXFORD PETS · LIVE DEMO</div>', unsafe_allow_html=True)
st.title("Meet the breed.")
st.caption("One pet. One photo. A little curiosity.")

left, right = st.columns([1.1, 1], gap="large")
with left:
    upload = st.file_uploader("Choose a pet photo", type=["jpg", "jpeg", "png", "webp"])
    example = st.segmented_control("Or try an example", list(EXAMPLES), default="Cat")
    if upload is not None:
        data = upload.getvalue()
    elif example is not None:
        data = (ASSETS / "examples" / EXAMPLES[example]).read_bytes()
    else:
        data = None
    image = None
    if data is not None:
        try:
            image = read_image(data)
            st.image(image, width="stretch")
        except ValueError as exc:
            st.error(str(exc))

with right:
    if image is None:
        st.info("Choose a photo to meet the breed.")
    elif st.button("Find the breed", type="primary", width="stretch"):
        try:
            with st.spinner("Taking a closer look…"):
                predictions = predict(image, predictor())
            st.markdown('<div class="result-label">OUR BEST GUESS</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="breed">{html.escape(predictions[0][0])}</div>', unsafe_allow_html=True)
            for index, (breed, probability) in enumerate(predictions):
                value = probability * 100
                muted = " muted" if index else ""
                st.markdown(
                    f'<div class="score{muted}"><span>{html.escape(breed)}</span><span>{value:.1f}%</span></div>'
                    f'<div class="track"><div class="fill" style="width:{value:.3f}%"></div></div>',
                    unsafe_allow_html=True,
                )
            st.caption("Model scores, not certainty. Similar breeds can look alike.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    else:
        st.markdown('<div class="result-label">READY WHEN YOU ARE</div>', unsafe_allow_html=True)
        st.write("A face, a few clues, a breed to discover.")
    st.caption("Knows 10 breeds. Other animals and breeds still receive a guess.")

with st.expander("The 10 breeds"):
    st.write("British Shorthair · Russian Blue · Siamese · Birman · American Bulldog · Pug · Sphynx · Maine Coon · Beagle · Samoyed")

st.markdown((ASSETS / "pets.html").read_text(encoding="utf-8"), unsafe_allow_html=True)
st.caption("Built with ResNet50 · Photos stay in memory for this session; the app does not save them.")
st.caption("Examples: Oxford-IIIT Pet / Parkhi et al. · [Source](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)")
