
import streamlit as st
import requests, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2, tensorflow as tf

st.set_page_config(page_title="Classification de Riz", page_icon="🌾", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F0F4F8; }
[data-testid="stSidebar"] { background: #1F4E79; }
[data-testid="stSidebar"] * { color: white !important; }
.bloc { background: white; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07); margin-bottom: 1rem; }
.titre { font-size: 2rem; font-weight: 800; color: #1F4E79; text-align: center; }
.sous-titre { font-size: 0.95rem; color: #666; text-align: center; margin-bottom: 1.5rem; }
.badge { background: #1F4E79; color: white; border-radius: 24px;
         padding: 0.5rem 1.4rem; font-size: 1.3rem; font-weight: 700; display: inline-block; }
.conf { font-size: 1rem; color: #2E75B6; font-weight: 600; margin-top: 0.4rem; }
.desc-box { background: #EAF3DE; border-left: 5px solid #3B6D11; border-radius: 0;
            padding: 0.8rem 1.2rem; color: #27500A; font-size: 0.95rem; margin: 0.5rem 0; }
.info-box { background: #E6F1FB; border-left: 5px solid #2E75B6; border-radius: 0;
            padding: 0.8rem 1.2rem; color: #1F4E79; font-size: 0.95rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

CLASSES  = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMG_SIZE = (224, 224)

# URL de l API deployee sur Render — sera mise a jour apres le deploiement
API_URL = "https://TON-API.onrender.com"

DESCRIPTIONS = {
    "Arborio"  : "Riz rond originaire d Italie, tres riche en amidon. Ideal pour le risotto.",
    "Basmati"  : "Riz long et parfume du sous-continent indien. Grains fins qui s allongent.",
    "Ipsala"   : "Variete turque a grains moyens, texture moelleuse apres cuisson.",
    "Jasmine"  : "Riz parfume de Thailande, grains longs legerement collants.",
    "Karacadag": "Variete turque a grains courts, tres populaire en cuisine locale.",
}

with st.sidebar:
    st.markdown("## 🌾 Classification de Riz")
    st.markdown("---")
    st.markdown("### Performances")
    c1, c2 = st.columns(2)
    c1.metric("Precision", "98.7%")
    c2.metric("F1-score", "98.7%")
    st.markdown("---")
    st.markdown("**Modele :** MobileNetV2")
    st.markdown("**Pre-entraine :** ImageNet")
    st.markdown("**Dataset :** 500 images")
    st.markdown("---")
    show_gc = st.checkbox("Afficher Grad-CAM", value=True)
    st.markdown("---")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.success("API connectee ✓") if r.ok else st.error("API non disponible")
    except:
        st.error("API non disponible")
    st.markdown("---")
    st.caption("Universite de Thies — UFR SET")
    st.caption("MaRT2 — Deep Learning 2025-2026")

st.markdown("<div class='titre'>🌾 Classification de Types de Riz</div>", unsafe_allow_html=True)
st.markdown("<div class='sous-titre'>Intelligence Artificielle · MobileNetV2 · Universite de Thies 2025-2026</div>", unsafe_allow_html=True)

col_g, col_d = st.columns([1, 1.4], gap="large")

with col_g:
    st.markdown("<div class='bloc'>", unsafe_allow_html=True)
    st.markdown("#### Image a analyser")
    fichier = st.file_uploader("Deposez une image de riz (JPG ou PNG)", type=["jpg","jpeg","png"])
    if fichier:
        img_pil = Image.open(fichier).convert("RGB")
        st.image(img_pil, use_column_width=True)
        analyser = st.button("🔍 Analyser", type="primary", use_container_width=True)
    else:
        st.markdown("<div class='info-box'>Deposez une image de grain de riz pour obtenir une prediction.</div>", unsafe_allow_html=True)
        analyser = False
    st.markdown("</div>", unsafe_allow_html=True)

with col_d:
    st.markdown("<div class='bloc'>", unsafe_allow_html=True)
    st.markdown("#### Resultats")
    if fichier and analyser:
        with st.spinner("Analyse en cours..."):
            fichier.seek(0)
            try:
                rep    = requests.post(f"{API_URL}/predict",
                           files={"file": (fichier.name, fichier, "image/jpeg")}, timeout=30)
                result = rep.json()
            except Exception as e:
                st.error(f"Erreur API : {e}") ; st.stop()
        cls  = result["classe"]
        conf = result["confiance"]
        prob = result["probabilites"]
        st.markdown(f"<div class='badge'>{cls}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='conf'>Confiance : {conf:.1f}%</div>", unsafe_allow_html=True)
        st.progress(int(conf))
        st.markdown(f"<div class='desc-box'><strong>{cls} :</strong> {DESCRIPTIONS[cls]}</div>", unsafe_allow_html=True)
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(6,3))
        fig.patch.set_facecolor("white") ; ax.set_facecolor("white")
        vals = [prob[c] for c in CLASSES]
        cols = ["#1F4E79" if c==cls else "#BDD7EE" for c in CLASSES]
        bars = ax.barh(CLASSES, vals, color=cols, edgecolor="white", height=0.55)
        for b,v in zip(bars,vals):
            ax.text(min(v+.5,105), b.get_y()+b.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=9, fontweight="600", color="#1F4E79")
        ax.set_xlim(0,115) ; ax.set_xlabel("Probabilite (%)") 
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        plt.tight_layout() ; st.pyplot(fig, use_container_width=True) ; plt.close()
        if show_gc:
            st.markdown("---")
            st.markdown("#### Grad-CAM")
            @st.cache_resource
            def load_m(): return tf.keras.models.load_model("modele_riz_final.h5")
            m   = load_m()
            arr = tf.cast(np.expand_dims(np.array(img_pil.resize(IMG_SIZE))/255.,0), tf.float32)
            idx = CLASSES.index(cls)
            xv  = tf.Variable(arr)
            with tf.GradientTape() as tape:
                tape.watch(xv)
                s = m(xv, training=False)[:,idx]
            g   = tape.gradient(s, xv)
            hm  = tf.reduce_mean(tf.abs(g[0]), axis=-1).numpy()
            hm  = hm / (hm.max()+1e-8)
            ia  = np.array(img_pil.resize(IMG_SIZE), dtype=np.uint8)
            hc  = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            sup = cv2.addWeighted(ia, 0.6, hc, 0.4, 0)
            c1,c2 = st.columns(2)
            c1.image(img_pil, caption="Originale", use_column_width=True)
            c2.image(Image.fromarray(sup), caption="Grad-CAM", use_column_width=True)
    elif not fichier:
        st.markdown("""<div style="text-align:center;padding:3rem;color:#aaa;">
            <div style="font-size:4rem">🌾</div>
            <div>Uploadez une image pour voir les resultats</div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.markdown("""<div style="text-align:center;font-size:0.8rem;color:#aaa;">
Projet Deep Learning · Classification de Types de Riz · Universite de Thies — UFR SET — MaRT2 · 2025-2026
</div>""", unsafe_allow_html=True)
