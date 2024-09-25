import numpy as np

def tile_to_latlng(TILE_SIZE, MIN_X_COORD, MIN_Y_COORD, tile_x, tile_y):
    latitude = tile_y * TILE_SIZE + MIN_Y_COORD
    longitude = tile_x * TILE_SIZE + MIN_X_COORD
    return latitude, longitude

def gen_tileranges(dataframe):
    tile_x_range = 0, dataframe[["tile_x"]].max().item()+1
    tile_y_range = 0, dataframe[["tile_y"]].max().item()+1
    return tile_x_range, tile_y_range

def map_temp(temp: float) -> int:
    if temp <= 0:
        return 0
    elif temp <= 10:
        return 1
    elif temp <= 20:
        return 2
    elif temp <= 28:
        return 3
    elif temp > 28:
        return 4
    else:
        raise ValueError("Received unexpected value: {}".format(temp))

def map_snow(snow: float) -> int:
    if snow == 0:
        return 0
    elif snow <= 10:
        return 1
    elif snow <= 30:
        return 2
    elif snow <= 50:
        return 3
    elif snow > 50:
        return 4
    else:
        raise ValueError("Received unexpected value: {}".format(snow))

def map_wind(wspd: float) -> int:
    if 0 <= wspd <= 5:
        return 0
    elif wspd <= 20:
        return 1
    elif wspd <= 40:
        return 2
    elif wspd <= 60:
        return 3
    elif wspd > 60:
        return 4
    else:
        raise ValueError("Received unexpected value: {}".format(wspd))

def map_coco(coco: float) -> int:
    if coco in [3.0, 4.0, 5.0, 7.0, 14.0, 17.0, 21.0]:
        return 1
    elif coco in [1.0, 2.0]:
        return 2
    elif coco in [6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 18.0, 19.0, 20.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]:
        return 0
    else:
        raise ValueError("Received unexpected value: {}".format(coco))

data_mappers = {
    "temp": map_temp,
    "snow": map_snow,
    "wspd": map_wind,
    "coco": lambda coco: map_coco(float([
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
    ].index(coco))+1.0)
}