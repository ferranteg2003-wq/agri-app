import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import json
import io
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="AgriApp - Gestione Progetti", layout="centered")
register_heif_opener()

# ==========================================
# CONNESSIONE A GOOGLE SHEETS
# ==========================================
def get_spreadsheet():
    url = st.secrets["URL_FOGLIO"]
    creds_dict = json.loads(st.secrets["CHIAVE_JSON"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, 
        ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(creds)
    return client.open_by_url(url)

# ==========================================
# MOTORE DI ANALISI IMMAGINI
# ==========================================
def analizza_cartina(uploaded_file, nome_personalizzato):
    image_bytes = uploaded_file.read()
    try:
        if uploaded_file.name.lower().endswith('.heic'):
            pil_image = Image.open(io.BytesIO(image_bytes))
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None, "Errore decodifica."
    except Exception as e: return None, str(e)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([50, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([160, 255, 255]))
    mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 90]))

    mask_cartina_fisica = cv2.bitwise_or(mask_yellow, mask_blue)
    mask_cartina_fisica = cv2.bitwise_or(mask_cartina_fisica, mask_dark)
    mask_cartina_fisica = cv2.morphologyEx(mask_cartina_fisica, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    contours, _ = cv2.findContours(mask_cartina_fisica, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, "Nessuna cartina individuata."

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    y1, y2 = max(0, y - 40), min(img.shape[0], y + h + 40)
    x1, x2 = max(0, x - 40), min(img.shape[1], x + w + 40)
    
    img_con_margine = img[y1:y2, x1:x2].copy()
    hsv_con_margine = hsv[y1:y2, x1:x2].copy()

    mask_forma_esatta = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cv2.drawContours(mask_forma_esatta, [max(contours, key=cv2.contourArea) - [x1, y1]], -1, 255, thickness=cv2.FILLED)

    mask_analisi = cv2.erode(mask_forma_esatta, np.ones((7,7), np.uint8), iterations=5)
    mask_giallo = cv2.inRange(hsv_con_margine, np.array([12, 60, 110]), np.array([50, 255, 255]))
    mask_bagnata = cv2.bitwise_and(cv2.bitwise_not(mask_giallo), cv2.bitwise_not(mask_giallo), mask=mask_analisi)

    pix_orig = max(1, cv2.countNonZero(mask_forma_esatta))
    pix_an = max(1, cv2.countNonZero(mask_analisi))
    area_reale = (pix_an / pix_orig) * 19.76
    
    pix_bag = cv2.countNonZero(mask_bagnata)
    perc_bag = (pix_bag / pix_an) * 100
    area_bag = (perc_bag / 100) * area_reale

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bagnata, connectivity=8)
    num_gocce = max(0, num_labels - 1)
    
    classi = {"micro":0, "piccole":0, "medie":0, "grandi":0, "extra":0}
    mm2_px = 1976.0 / pix_orig
    for i in range(1, num_labels):
        area_mm2 = stats[i, cv2.CC_STAT_AREA] * mm2_px
        if area_mm2 < 0.1: classi["micro"]+=1
        elif area_mm2 < 0.5: classi["piccole"]+=1
        elif area_mm2 < 1.0: classi["medie"]+=1
        elif area_mm2 < 2.0: classi["grandi"]+=1
        else: classi["extra"]+=1

    img_rosse = img_con_margine.copy()
    img_rosse[mask_bagnata > 0] = [0, 0, 255]

    dati = {
        "file": nome_personalizzato, "area_reale": area_reale, "bagnatura": perc_bag,
        "area_bag": area_bag, "gocce": num_gocce, "densita": num_gocce/area_reale,
        "pulita": 100 - perc_bag, "dist": classi
    }
    return (cv2.cvtColor(img_rosse, cv2.COLOR_BGR2RGB), dati), None

# ==========================================
# INTERFACCIA WEB (STREAMLIT)
# ==========================================
st.title("💧 AgriApp - Area Progetti")

if "URL_FOGLIO" not in st.secrets: 
    st.warning("⚠️ Manca la configurazione dei Secrets!")
    st.stop()

# 1. Carica il documento e tutti i progetti (fogli) esistenti
try:
    sh = get_spreadsheet()
    fogli_esistenti = sh.worksheets()
    nomi_progetti = [f.title for f in fogli_esistenti]
except Exception as e:
    st.error(f"Impossibile collegarsi al database: {e}")
    st.stop()

st.markdown("### 📁 Gestione Progetti")
# Aggiungiamo le opzioni speciali al menù a tendina
opzioni_menu = ["-- Seleziona un progetto --", "➕ Crea Nuovo Progetto..."] + nomi_progetti
progetto_scelto = st.selectbox("Scegli il progetto su cui lavorare:", opzioni_menu)

# ==========================================
# LOGICA: CREA NUOVO PROGETTO
# ==========================================
if progetto_scelto == "➕ Crea Nuovo Progetto...":
    st.info("Creando un nuovo progetto verrà generata una nuova scheda nel tuo file Excel originale, mantenendo intatte le tue intestazioni.")
    nuovo_nome = st.text_input("Inserisci il nome del nuovo progetto:")
    
    if st.button("Genera Progetto"):
        if nuovo_nome in nomi_progetti:
            st.error("Esiste già un progetto con questo nome! Scegline un altro.")
        elif nuovo_nome == "":
            st.warning("Il nome non può essere vuoto.")
        else:
            with st.spinner("Creazione del progetto in corso..."):
                try:
                    # Duplica il primo foglio (che funge da template originale)
                    nuovo_foglio = sh.duplicate_sheet(fogli_esistenti[0].id, new_sheet_name=nuovo_nome)
                    
                    # Svuota i dati vecchi copiati per sbaglio, lasciando intatta la Riga 1 (Intestazione)
                    if nuovo_foglio.row_count > 1:
                        nuovo_foglio.delete_rows(2, nuovo_foglio.row_count)
                        
                    st.success(f"Progetto '{nuovo_nome}' creato con successo!")
                    st.rerun() # Riavvia l'app per aggiornare il menù a tendina
                except Exception as e:
                    st.error(f"Errore durante la creazione: {e}")

# ==========================================
# LOGICA: PROGETTO ESISTENTE (Rinomina & Analisi)
# ==========================================
elif progetto_scelto != "-- Seleziona un progetto --":
    
    # --- SEZIONE RINOMINA ---
    with st.expander("✏️ Rinomina questo progetto"):
        nuovo_nome_rinomina = st.text_input("Nuovo nome:", value=progetto_scelto)
        if st.button("Salva Modifica"):
            if nuovo_nome_rinomina != progetto_scelto:
                if nuovo_nome_rinomina in nomi_progetti:
                    st.error("Esiste già un progetto con questo nome.")
                else:
                    with st.spinner("Rinominando..."):
                        foglio_da_modificare = sh.worksheet(progetto_scelto)
                        foglio_da_modificare.update_title(nuovo_nome_rinomina)
                        st.success("Nome aggiornato!")
                        st.rerun()
    
    st.write("---")
    st.markdown(f"### 📷 Acquisizione Dati: **{progetto_scelto}**")
    
    # --- SEZIONE ANALISI E FOTO ---
    files = st.file_uploader("Scatta o carica cartine", type=['jpg', 'jpeg', 'png', 'heic'], accept_multiple_files=True)

    if files:
        st.markdown("📝 **Rinomina le tue acquisizioni prima del salvataggio:**")
        nomi_personalizzati = {}
        for f in files:
            nome_inserito = st.text_input(f"Nome per: {f.name}", value=f.name, key=f.name)
            nomi_personalizzati[f.name] = nome_inserito
            
        if st.button(f"🚀 Analizza e Salva in '{progetto_scelto}'"):
            foglio_destinazione = sh.worksheet(progetto_scelto) # Selezioniamo il foglio corretto
            
            for f in files:
                nome_da_salvare = nomi_personalizzati[f.name]
                st.write("---")
                st.subheader(f"📄 Elaborazione: {nome_da_salvare}")
                
                res, err = analizza_cartina(f, nome_da_salvare)
                if err: 
                    st.error(err)
                else:
                    img, dati = res
                    col1, col2 = st.columns([1,1])
                    col1.image(img, use_container_width=True)
                    with col2:
                        st.success(f"Bagnatura: {dati['bagnatura']:.1f}%")
                        st.info(f"Densità: {dati['densita']:.1f} g/cm²")
                    
                    with st.spinner("Salvataggio nel progetto..."):
                        try:
                            riga = [
                                dati['file'], "76 x 26", dati['area_reale'], dati['bagnatura']/100.0, dati['area_bag'],
                                dati['gocce'], dati['densita'], dati['pulita']/100.0,
                                dati['dist']['micro'], dati['dist']['piccole'], dati['dist']['medie'], dati['dist']['grandi'], dati['dist']['extra']
                            ]
                            # Salviamo specificatamente nel foglio scelto
                            foglio_destinazione.append_row(riga, value_input_option="USER_ENTERED")
                            st.toast(f"✅ Salvato: {nome_da_salvare}", icon="☁️")
                        except Exception as e:
                            st.error(f"Errore di salvataggio: {e}")
            
            st.balloons()
