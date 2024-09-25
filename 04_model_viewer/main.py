from model import MultinomialNBClassifier
from helpers import tile_to_latlng, data_mappers
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import pydeck as pdk
import os

# Konfigurieren von der Tile - Eigenschaften

MIN_X_COORD = 12.36749421446289
MAX_X_COORD = 14.312163310124404

MIN_Y_COORD = 51.948449733535
MAX_Y_COORD = 52.978667577725275
TILE_SIZE = 0.01

# Beobachte (Trainings)Daten einlesen und Intervall gültiger Tiles generieren, falls ein Modell noch nicht generiert wurde
if not os.path.isfile('../data/model.pkl'):
    print("No model was found. Building new one from Dataset...")

    df = pd.read_csv('../data/synth_access_data.csv')
    print("Read CSV")
    mnbc = MultinomialNBClassifier(df, (0, 195), (0, 104), {
        'temp': [0, 1, 2, 3, 4],
        'snow': [0, 1, 2, 3, 4],
        'wspd': [0, 1, 2, 3, 4],
        'coco': [0, 1, 2],
        'vacation': [0, 1],
        'holiday': [0, 1],
        'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'weekday': [0, 1, 2, 3, 4, 5, 6],
        'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    })
    
    print("Saving model @ '{}'".format('../data/model.pkl'))
    mnbc.save('../data/model.pkl')

# Generieren des probabilistischen Modells basierend auf den Daten
if "model" not in st.session_state:
    st.session_state.model = MultinomialNBClassifier.load('../data/model.pkl')
query_parameters = {}

st.markdown("## Visualisierung des Modells")

temp_checked = st.sidebar.checkbox("Temperatur auswählen")
if temp_checked:
    temp = st.sidebar.slider(
        "Temperatur (in °C)",
        -5, 30,
        value=20
    )

    query_parameters["temp"] = data_mappers["temp"](float(temp))


snow_checked = st.sidebar.checkbox("Schneehöhe auswählen")
if snow_checked:
    snow = st.sidebar.slider(
        "Schneehöhe (in cm)",
        0, 55,
        value=0
    )

    query_parameters["snow"] = data_mappers["snow"](float(snow))


wspd_checked = st.sidebar.checkbox("Windgeschwindigkeit auswählen")
if wspd_checked:
    wspd = st.sidebar.slider(
        "Windgeschwindigkeit (in km/h)",
        0, 60
    )

    query_parameters["wspd"] = data_mappers["wspd"](float(wspd))


coco_checked = st.sidebar.checkbox("Wetterverhältnis auswählen")
if coco_checked:
    coco = st.sidebar.selectbox(
        "Wetterverhältnis",
        options=[
            "Klar",
            "Heiter",
            "Bewölkt",
            "Bedeckt",
            "Nebel",
            "Gefrierender Nebel",
            "Leichter Regen",
            "Regen",
            "Starker Regen",
            "Gefrierender Regen",
            "Starker gefrierender Regen",
            "Schneeregen",
            "Starker Schneeregen",
            "Leichter Schneefall",
            "Schneefall",
            "Starker Schneefall",
            "Regenschauer",
            "Starker Regenschauer",
            "Schneeregenschauer",
            "Starker Schneeregenschauer",
            "Schneeschauer",
            "Starker Schneeschauer",
            "Blitze",
            "Hagel",
            "Gewitter",
            "Starkes Gewitter",
            "Sturm"
        ]
    )
    
    if coco:
        query_parameters["coco"] = data_mappers["coco"](coco)


vacation_checked = st.sidebar.checkbox("Ferien auswählen")
if vacation_checked:
    vacation = st.sidebar.selectbox(
        "Ferienstatus",
        options=[
            "Ferien",
            "Keine Ferien"
        ]
    )

    if vacation:
        query_parameters["vacation"] = int(vacation == "Ferien")


holiday_checked = st.sidebar.checkbox("Feiertag auswählen")
if holiday_checked:
    holiday = st.sidebar.selectbox(
        "Feiertagsstatus",
        options=[
            "Feiertag",
            "Kein Feiertag"
        ]
    )

    if holiday:
        query_parameters["holiday"] = int(holiday=="Feiertag")


date_checked = st.sidebar.checkbox("Datum auswählen")
if date_checked:
    date = st.sidebar.date_input(
        "Datum",
        min_value=datetime.datetime(2023, 1, 1),
        max_value=datetime.datetime(2023, 12, 31),
        format="DD.MM.YYYY"
    )

    day = date.day
    query_parameters["month"] = date.month
    query_parameters["day"] = date.day


hour_checked = st.sidebar.checkbox("Zeit auswählen")
if hour_checked:
    time_selected = st.sidebar.time_input(
        "Zeit",
        value=datetime.time(12, 0, 0),
        step=3600
    )

    query_parameters["hour"] = time_selected.hour

tile_probabilities = st.session_state.model.predict(query_parameters)

st.markdown("### Karte")
latlngs = [tile_to_latlng(TILE_SIZE, MIN_X_COORD, MIN_Y_COORD, *tile) for tile in tile_probabilities.keys()]
lat, lng = zip(*latlngs)

df = pd.DataFrame({'lat': lat, 'lng': lng, 'prob': list(tile_probabilities.values())})
st.pydeck_chart(pdk.Deck(layers=[pdk.Layer(
    'GridLayer',
    df,
    get_position=['lng', 'lat'],
    auto_highlight=True,
    elevation_scale=50,
    pickable=False,
    get_color_weight='prob',
    get_elevation_weight='prob',
    extruded=True,
    cell_size=500,
    coverage=1
)], initial_view_state=pdk.ViewState(
    longitude=MIN_X_COORD + (MAX_X_COORD-MIN_X_COORD)/2,
    latitude=MAX_Y_COORD + (MAX_Y_COORD-MIN_Y_COORD)/2,
    zoom=10,
    min_zoom=5,
    max_zoom=6,
    pitch=45,
    bearing=9#-27.36
)))

st.markdown("### Parameter:")
st.write(query_parameters)

st.markdown("### Top Tiles:")
n_samples = st.slider("Anzahl der Tiles",
    1, min(100, len(tile_probabilities))
)
l = [*sorted([[tile, probability] for tile, probability in tile_probabilities.items()],
             key=lambda v: v[1], reverse=True)][:n_samples]
st.write([{'tile_position': e[0], 'probability': e[1]} for e in l])
