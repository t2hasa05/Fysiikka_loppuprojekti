import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import plotly.express as px

path1 = "https://raw.githubusercontent.com/t2hasa05/Fysiikka_loppuprojekti/refs/heads/main/Data/Loppuprojekti_Sijainti.csv"
path2 = "https://raw.githubusercontent.com/t2hasa05/Fysiikka_loppuprojekti/refs/heads/main/Data/Loppuprojekti_Kiihtyvyys.csv"
df_s = pd.read_csv(path1)
df_k = pd.read_csv(path2)

# Rajataan datasta pois ne rivit, joilla horisontaalinen (epä)tarkkuus on suuri
df_s = df_s[df_s['Horizontal Accuracy (m)'] <10] # Rajataan pois arvot, joissa Horizontal Accuracy (m) >10
df_s = df_s.reset_index(drop=True)

st.title('Tietoja päivän liikunnasta')

## Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta:

# Tuodaan filtterifunktiot
from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff,  nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

# Data suodatetaan alipäästösuodattimella, joka poistaa siitä valittua cut-off -taajuutta suuremmilla taajuuksilla tapahtuvat vaihtelut
# Käytännössä dataa "tasoitetaan", eli alipäästösuodatin vastaa jossain määrin liukuvaa keskiarvoa

data = df_k['Z (m/s^2)']
T_tot = df_k['Time (s)'].max() # Koko datan pituus
n = len(df_k['Time (s)']) # Datapisteiden lukumäärä
fs = n/T_tot # Näytteenottotaajuus (oletetaan vakioksi)
nyq = fs/2 # Nyqvistin taajuus, suurin taajuus joka datasta voidaan havaita
order = 3
cutoff = 1/0.2 # Cut-off -taajuus, tätä suuremmat taajuudet alipäästösuodatin poistaa datasta
data_filt = butter_lowpass_filter(data, cutoff, nyq, order)
df_k['Z_filtered (m/s^2)'] = data_filt

# Lasketaan askelet
# Tutkitaan, kuinka usein suodatettu signaali ylittää nollatason
jaksot = 0
for i in range (n-1):
    if data_filt[i]/data_filt[i+1] < 0:
        jaksot = jaksot + 1/2

st.write("Askelmäärä (laskettuna suodatetusta kiihtyvyysdatasta): ", round(jaksot), " askelta")

## Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella:

# Valittiin kiihtyvyyden z-komponentti

signal = df_k['Z (m/s^2)']
t = df_k['Time (s)'] # Aika alkaa nollasta, sekunteina
N = len(signal) # Havaintojen määrä
dt = np.max(t)/N # Näytteenottoväli (oletetaan vakioksi)

# Fourier-analyysi
fourier = np.fft.fft(signal,N) # Fourier-muunnos
psd = fourier*np.conj(fourier)/N # Tehospektri
freq = np.fft.fftfreq(N,dt) # Taajuudet
L = np.arange(1,int(N/2)) # Negatiivisten ja nollataajuuksien rajaus pois

f_max = freq[L][psd[L] == np.max(psd[L])][0] # Askelluksen dominoiva taajuus
T = 1/f_max # Jaksonaika
steps = f_max*np.max(t) # Askelmäärä
st.write("Askelmäärä (Fourier-analyysin perusteella): ", round(steps), " askelta")

## Kuljettu matka (GPS-datasta)

#Lasketaan matka käyttäen Haversinen kaava
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

# Lasketaan kuljettu matka
import numpy as np
df_s['Distance_calc'] = np.zeros(len(df_s))

# Lasketaan välimatka havaintopisteiden välillä käyttäen for-looppia
for i in range(len(df_s)-1):
    lon1 = df_s['Longitude (°)'][i]
    lon2 = df_s['Longitude (°)'][i+1]
    lat1 = df_s['Latitude (°)'][i]
    lat2 = df_s['Latitude (°)'][i+1]
    df_s.loc[i+1,'Distance_calc'] = haversine(lon1, lat1, lon2, lat2)

## Kuljettu matka (GPS-datasta)
df_s['total_distance'] = df_s['Distance_calc'].cumsum() # Kumulatiivinen matka
total_distance = df_s['Distance_calc'].sum() # Kokonaismatka
total_distance_m = total_distance * 1000 # Muunnetaan metreiksi
st.write("Kuljettu matka: ", round(total_distance_m), " m")

## Keskinopeus (GPS-datasta)
total_time_s = df_s['Time (s)'].iloc[-1] - df_s['Time (s)'].iloc[0]
average_speed = total_distance_m / total_time_s
st.write("Keskinopeus: ", round(average_speed, 1), " m/s")

## Askelpituus (lasketun askelmäärän ja matkan perusteella)
askelmaara = jaksot # Suodatetusta kiihtyvyysdatasta laskettu askelmäärä
askelpituus = total_distance_m / askelmaara
st.write("Askelpituus: ", round(askelpituus*100), " cm")

## Suodatettu kiihtyvyysdata (valittu z-komponentti)
st.title('Suodatetun kiihtyvyysdatan z-komponentti')
fig = px.line(
    df_k,
    x='Time (s)',
    y='Z_filtered (m/s^2)',
    labels={
        'Time (s)': 'Aika (s)',
        'Z_filtered (m/s^2)': 'Suodatettu kiihtyvyys (m/s²)'
    }
)
st.plotly_chart(fig, width='stretch')

## Analyysiin valitun kiihtyvyysdatan komponentin tehospektritiheys
st.title('Tehospektri')
fig, ax = plt.subplots(figsize=(15,6))

ax.plot(freq[L], psd[L].real)
ax.grid()
ax.set_xlabel('Taajuus [Hz]')
ax.set_ylabel('Teho')
ax.set_xlim(0, 10)
ax.set_ylim(0, 11000)

st.pyplot(fig)

## Reitti kartalla
st.title('Karttakuva')

# Määritellään "karttapohja" eli kartan keskipiste:
lat1 = df_s['Latitude (°)'].mean()
long1 = df_s['Longitude (°)'].mean()

# Luodaan kartta
my_map = folium.Map(location = [lat1,long1], zoom_start=17)

# Piirretään reitti kartalle:
folium.PolyLine(df_s[['Latitude (°)','Longitude (°)']], color = 'red', weight = 3).add_to(my_map)
my_map.save('Kartta_18.12.2025.html')
st_map = st_folium(my_map, width = 900, height = 650)