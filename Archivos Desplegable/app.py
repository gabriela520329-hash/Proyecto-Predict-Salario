"""
=============================================================
  Predictor de Salario - Proyecto Analítica de Datos
  Universidad Icesi | Ingeniería Industrial
=============================================================
  Para correr la app localmente:
      pip install streamlit tensorflow joblib numpy
      streamlit run app.py
=============================================================
"""
import streamlit as st
import numpy as np
import joblib
import os

# ─── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Salario - BN-MLP",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Estilos CSS personalizados ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Outfit:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* ── Fondo general con textura sutil ── */
    .stApp {
        background-color: #080c12;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(226,185,111,0.08) 0%, transparent 70%),
            url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23e2b96f' fill-opacity='0.02'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        min-height: 100vh;
    }

    /* ── Contenedor principal más angosto y centrado ── */
    .block-container {
        max-width: 760px !important;
        padding: 3rem 2rem 4rem !important;
    }

    /* ── Hero ── */
    .hero-eyebrow {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #e2b96f;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 0.6rem;
        opacity: 0.9;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #f5e9cf;
        text-align: center;
        line-height: 1.05;
        margin-bottom: 0.8rem;
        letter-spacing: -0.01em;
    }
    .hero-title span {
        background: linear-gradient(90deg, #e2b96f 0%, #f5d78e 50%, #e2b96f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #6b7a8d;
        text-align: center;
        margin-bottom: 3rem;
        line-height: 1.6;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
    }

    /* ── Divisor decorativo ── */
    .divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.4rem;
    }
    .divider-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(226,185,111,0.2), transparent);
    }
    .divider-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        color: #e2b96f;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        white-space: nowrap;
        opacity: 0.85;
    }

    /* ── Tarjetas ── */
    .card {
        background: linear-gradient(145deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.015) 100%);
        border: 1px solid rgba(226,185,111,0.1);
        border-top: 1px solid rgba(226,185,111,0.22);
        border-radius: 20px;
        padding: 1.8rem 2rem 2rem;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(226,185,111,0.4), transparent);
    }

    /* ── Resultado ── */
    .result-box {
        background: linear-gradient(145deg, rgba(226,185,111,0.09) 0%, rgba(226,185,111,0.03) 100%);
        border: 1px solid rgba(226,185,111,0.3);
        border-radius: 20px;
        padding: 2.8rem 2rem;
        text-align: center;
        margin-top: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .result-box::after {
        content: '';
        position: absolute;
        bottom: -40px; right: -40px;
        width: 140px; height: 140px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(226,185,111,0.08) 0%, transparent 70%);
    }
    .result-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #8892a4;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .result-salary {
        font-family: 'Playfair Display', serif;
        font-size: 4.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e2b96f 0%, #f5d78e 50%, #c9a050 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .result-period {
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem;
        font-weight: 400;
        color: #6b7a8d;
        letter-spacing: 0.04em;
    }
    .result-period strong {
        color: #a0aab8;
        font-weight: 500;
    }

    /* ── Badges ── */
    .badges-wrap {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 1.8rem;
    }
    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(226,185,111,0.07);
        border: 1px solid rgba(226,185,111,0.18);
        color: #c8a86b;
        border-radius: 100px;
        padding: 0.28rem 0.9rem;
        font-size: 0.78rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
    }

    /* ── Alertas ── */
    .warning-box {
        background: rgba(255,180,50,0.06);
        border-left: 3px solid rgba(255,180,50,0.5);
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.3rem;
        color: #d4a84b;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }
    .error-box {
        background: rgba(255,80,80,0.06);
        border-left: 3px solid rgba(255,80,80,0.4);
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.3rem;
        color: #e07070;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.04);
    }
    .footer-project {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #e2b96f;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 0.4rem;
    }
    .footer-names {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        color: #3d4654;
        font-weight: 300;
    }

    /* ── Overrides de Streamlit ── */
    .stSelectbox label,
    .stSlider label,
    .stNumberInput label {
        color: #8892a4 !important;
        font-size: 0.83rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #d8e0ea !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Slider dorado */
    .stSlider > div > div > div {
        background: rgba(226,185,111,0.2) !important;
    }
    div[data-testid="stSlider"] > div > div > div > div {
        background: #e2b96f !important;
        box-shadow: 0 0 0 3px rgba(226,185,111,0.2) !important;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #c9982a 0%, #e2b96f 40%, #f0cc80 100%) !important;
        color: #0c0e14 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
        box-shadow: 0 4px 20px rgba(226,185,111,0.15) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 32px rgba(226,185,111,0.28) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Cargar modelo y artefactos del BN-MLP ────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    archivos = {
        'modelo':   'modelo_salario.keras',
        'scaler':   'scaler_salario.pkl',
        'encoders': 'encoders_salario.pkl'
    }
    faltantes = [nombre for nombre, archivo in archivos.items() if not os.path.exists(archivo)]
    if faltantes:
        return None, None, None, faltantes
    try:
        import tensorflow as tf
        modelo   = tf.keras.models.load_model(archivos['modelo'])
        scaler   = joblib.load(archivos['scaler'])
        encoders = joblib.load(archivos['encoders'])
        return modelo, scaler, encoders, []
    except Exception as e:
        return None, None, None, [str(e)]

modelo, scaler, encoders, errores = cargar_modelo()

# ─── Categorías ──────────────────────────────────────────────────────────────
WORK_TYPES      = ['Contract', 'Full-Time', 'Intern', 'Part-Time', 'Temporary']
QUALIFICATIONS  = ['BA', 'BBA', 'BCA', 'B.Com', 'B.Tech', 'MBA', 'MCA', 'M.Com', 'M.Tech', 'PhD']
COUNTRIES_TOP   = [
    'Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Australia',
    'Austria', 'Belgium', 'Brazil', 'Canada', 'Chile', 'China',
    'Colombia', 'Czech Republic', 'Denmark', 'Egypt', 'Finland',
    'France', 'Germany', 'Greece', 'Hungary', 'India', 'Indonesia',
    'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan',
    'Kenya', 'Malaysia', 'Mexico', 'Morocco', 'Netherlands',
    'New Zealand', 'Nigeria', 'Norway', 'Pakistan', 'Peru',
    'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia',
    'Saudi Arabia', 'South Africa', 'South Korea', 'Spain', 'Sweden',
    'Switzerland', 'Thailand', 'Turkey', 'Ukraine', 'United Kingdom',
    'United States', 'Venezuela', 'Vietnam'
]
ROLES_TOP = [
    'Account Manager', 'Business Analyst', 'Cloud Architect',
    'Customer Success Manager', 'Data Analyst', 'Data Engineer',
    'Data Scientist', 'DevOps Engineer', 'Financial Analyst',
    'Frontend Developer', 'Full Stack Developer', 'HR Manager',
    'Interaction Designer', 'Marketing Manager', 'Network Administrator',
    'Operations Manager', 'Product Manager', 'Project Manager',
    'Sales Manager', 'Software Engineer', 'System Administrator',
    'UI/UX Designer', 'User Interface Designer'
]

def get_clases(nombre, fallback):
    if encoders and nombre in encoders:
        return list(encoders[nombre].classes_)
    return fallback

# ─── Predicción ───────────────────────────────────────────────────────────────
def predecir_salario(experience, company_size, latitude, longitude,
                     year, work_type, qualification, country, role):
    def encode(col_name, valor, fallback_list):
        if encoders and col_name in encoders:
            le = encoders[col_name]
            if valor in le.classes_:
                return int(le.transform([valor])[0])
            else:
                return len(le.classes_) // 2
        else:
            if valor in fallback_list:
                return fallback_list.index(valor)
            return 0

    wt_enc   = encode('Work Type',      work_type,     WORK_TYPES)
    qual_enc = encode('Qualifications', qualification, QUALIFICATIONS)
    co_enc   = encode('Country',        country,       COUNTRIES_TOP)
    role_enc = encode('Role',           role,          ROLES_TOP)

    X_input = np.array([[
        experience, company_size, latitude, longitude,
        year, wt_enc, qual_enc, co_enc, role_enc
    ]])
    X_scaled = scaler.transform(X_input)
    pred = modelo.predict(X_scaled, verbose=0)
    return float(pred[0][0])

# ═══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-eyebrow">Red neuronal BN-MLP · Universidad Icesi</p>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title"><span>Salary</span> Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ingresa las características del cargo para estimar el salario promedio usando inteligencia artificial</p>', unsafe_allow_html=True)

# ── Alerta si faltan archivos ──────────────────────────────────────────────────
if errores:
    st.markdown(f"""
    <div class="warning-box">
        ⚠ <strong>Archivos del modelo no detectados.</strong><br>
        Asegúrate de que <code>modelo_salario.keras</code>, <code>scaler_salario.pkl</code> y 
        <code>encoders_salario.pkl</code> estén en la raíz del repositorio.<br>
        <em style="opacity:0.7;">Detalle: {', '.join(errores)}</em>
    </div>
    """, unsafe_allow_html=True)

# ── Tarjeta 1: Experiencia & Empresa ─────────────────────────────────────────
st.markdown("""
<div class="card">
  <div class="divider">
    <div class="divider-line"></div>
    <span class="divider-label">🏢 &nbsp;Experiencia &amp; Empresa</span>
    <div class="divider-line"></div>
  </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    experience = st.slider(
        "Años de experiencia requeridos",
        min_value=0, max_value=20, value=3, step=1,
        help="Experiencia laboral mínima solicitada por la vacante"
    )
with col2:
    company_size = st.number_input(
        "Tamaño de la empresa (N° empleados)",
        min_value=10, max_value=500000, value=5000, step=500,
        help="Volumen total de empleados dentro de la organización"
    )

col3, col4 = st.columns(2)
with col3:
    qualification = st.selectbox(
        "Nivel educativo exigido",
        options=get_clases('Qualifications', QUALIFICATIONS)
    )
with col4:
    work_type = st.selectbox(
        "Tipo de jornada laboral",
        options=get_clases('Work Type', WORK_TYPES)
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Tarjeta 2: Cargo & Ubicación ──────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div class="divider">
    <div class="divider-line"></div>
    <span class="divider-label">📍 &nbsp;Cargo &amp; Ubicación</span>
    <div class="divider-line"></div>
  </div>
""", unsafe_allow_html=True)

role = st.selectbox(
    "Rol profesional / Cargo",
    options=get_clases('Role', ROLES_TOP)
)

col5, col6 = st.columns(2)
with col5:
    country = st.selectbox(
        "País de contratación",
        options=get_clases('Country', COUNTRIES_TOP)
    )
with col6:
    year = st.selectbox(
        "Año de publicación de la oferta",
        options=list(range(2019, 2027)),
        index=4
    )

col7, col8 = st.columns(2)
with col7:
    latitude = st.number_input(
        "Latitud geográfica",
        min_value=-90.0, max_value=90.0, value=4.7110,
        format="%.4f",
        help="Coordenada decimal norte-sur del lugar de empleo"
    )
with col8:
    longitude = st.number_input(
        "Longitud geográfica",
        min_value=-180.0, max_value=180.0, value=-74.0721,
        format="%.4f",
        help="Coordenada decimal este-oeste del lugar de empleo"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Botón de predicción ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("✦ Calcular Salario Estimado")

if predict_btn:
    if modelo is None:
        st.markdown("""
        <div class="error-box">
            ✕ <strong>Error de Ejecución:</strong> Los pesos de la red neuronal no han sido cargados.
            Comprueba las rutas del repositorio.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Procesando a través de las capas BN-MLP…"):
            try:
                salario = predecir_salario(
                    experience, company_size, latitude, longitude,
                    year, work_type, qualification, country, role
                )
                salario_mes = salario / 12

                st.markdown(f"""
                <div class="result-box">
                    <p class="result-label">Salario Anual Estimado</p>
                    <p class="result-salary">${salario:,.0f}</p>
                    <p class="result-period">
                        <strong>${salario_mes:,.0f}</strong> por mes &nbsp;·&nbsp; USD
                    </p>
                </div>
                <div class="badges-wrap">
                    <span class="info-badge">🎓 {qualification}</span>
                    <span class="info-badge">💼 {work_type}</span>
                    <span class="info-badge">📌 {role}</span>
                    <span class="info-badge">🌍 {country}</span>
                    <span class="info-badge">⏱ {experience} años exp.</span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ✕ <strong>Error de Inferencia:</strong> <code>{str(e)}</code>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p class="footer-project">Proyecto Analítica de Datos · Universidad Icesi · 2026</p>
    <p class="footer-names">Angely Palomeque · Gabriela Bolaños · Mariana Jiménez · Haider Basante</p>
</div>
""", unsafe_allow_html=True)
