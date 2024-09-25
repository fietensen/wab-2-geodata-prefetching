from sqlalchemy import create_engine
import pandas as pd
import shapely as shp
import numpy as np
import collections

query = """
SELECT geom, aeroway, amenity, building, capacity, "isced_level", leisure, name, opening_hours, shop, tourism
FROM poi__points_of_interest
WHERE shop IS NOT NULL
   OR amenity='marketplace' OR shop='convenience' OR shop='supermarket'
   OR amenity IN ('restaurant', 'fast_food', 'cafe', 'bar', 'pub')
   OR tourism IS NOT NULL
   OR amenity IN ('kindergarten', 'school', 'college', 'university', 'language_school')
   OR amenity='toilets'
   OR leisure='swimming_pool'
   OR aeroway IS NOT NULL OR building='aerodrome';
"""

class HitrateGenerator:
    def __init__(self, engine_uri):
        self.__engine = create_engine(engine_uri)
        self.__poi_df = pd.read_sql(query, self.__engine)

        self.__poi_tiles = []
        for i, row in self.__poi_df.iterrows():
            xy_p = shp.from_wkb(row['geom']).centroid.xy
            x, y = xy_p[0][0], xy_p[1][0]
            tx, ty = lonlat_to_tile(x, y, 0.01)
            tx = max(min(tx, MAX_TILE[0]), MIN_TILE[0])
            ty = max(min(ty, MAX_TILE[1]), MIN_TILE[1])

            self.__poi_tiles.append((tx, ty))


    def _hit_generator(self, tile_visit_rates):
        while True:
            tile_hits = {}

            for tile, visit_rate in tile_visit_rates.items():
                c_tile_hits = np.random.poisson(lam=visit_rate)
                if c_tile_hits:
                    tile_hits[tile] = c_tile_hits
        
            tiles, _ = zip(*sorted(tile_hits.items(), key=lambda i: i[1], reverse=True))
        
            # Return 50% of most requested requested tiles
            yield tiles[:len(tiles)//2]


    def make_hit_gen(self, ext_factors):
        tile_visit_rates = collections.defaultdict(float)
        for j, poi in self.__poi_df.iterrows():
            tile_visit_rates[self.__poi_tiles[j]] += calculate_poi_prate(poi, ext_factors)
        
        return self._hit_generator(tile_visit_rates)


def educational_rate(poi, efactors):
    if efactors['weekday'] > 5 or efactors['vacation'] or efactors['holiday']:
        return 0

    rate = 2.0

    if efactors['snow'] >= 3:
        rate -= 0.2
    
    if efactors['temp'] == 0 or efactors['temp'] == 4:
        rate -= 0.2
    
    if 7 <= efactors['hour'] <= 9:
        rate += 20.0

    return max(0.1, rate)


def leisure_rate(poi, efactors):
    outdoors = poi['leisure'] in [
        'picnic_table',
        'garden',
        'swimming_pool',
        'horse_riding',
        'bird_hide',
        'playground',
        'wildlife_hide',
        'camping',
        'park',
        'maze',
        'beach_resort',
        'outdoor_seating']
    
    rate = 2.0

    if outdoors and efactors['temp'] <= 1 or efactors['coco'] == 0 or efactors['snow'] > 1 or efactors['wspd'] >= 3:
        rate -= 10.0
    elif not outdoors:
        rate += 2.0

    if efactors['holiday'] and efactors['vacation'] and 10 <= efactors['hour'] <= 21:
        rate += 10.0
    elif (efactors['holiday'] or efactors['vacation'] or efactors['weekday'] > 5) and 10 <= efactors['hour'] <= 21:
        rate += 10.0
    elif 12 <= efactors['hour'] <= 14 or 17 <= efactors['hour'] <= 21:
        rate += 10.0

    return max(0.1, rate)


def commercial_rate(poi, efactors):
    essential = poi['shop'] in ['convenience', 'supermarket', 'grocery', 'mall']

    rate = 2.0

    if poi['shop'] != None and efactors['weekday'] == 7:
        return 0.0
    elif poi['shop'] != None and efactors['weekday'] == 6:
        rate += 15.0
    elif poi['shop'] != None and 17 <= efactors['hour'] <= 19:
        rate += 10.0


    if poi['shop'] == None and 6 <= efactors['month'] <= 8:
        rate += 20.0
    elif poi['shop'] == None:
        rate += 5.0

    return max(0.1, rate)


def calculate_poi_prate(poi, efactors):
    if poi['isced_level'] != None:
        return educational_rate(poi, efactors)
    elif poi['leisure'] != None:
        return leisure_rate(poi, efactors)
    elif (poi['shop'] != None and poi['shop'] != 'no') or poi['tourism'] != None:
        return commercial_rate(poi, efactors)
    else:
        return 0.1
    
def lonlat_to_tile(lon, lat, tile_size):
    x_tile = int((lon - MIN_X_COORD) / tile_size)
    y_tile = int((lat - MIN_Y_COORD) / tile_size)
    return x_tile, y_tile

MIN_X_COORD = 12.36749421446289
MAX_X_COORD = 14.312163310124404

MIN_Y_COORD = 51.948449733535
MAX_Y_COORD = 52.978667577725275

MIN_TILE = lonlat_to_tile(MIN_X_COORD, MIN_Y_COORD, 0.01)
MAX_TILE = lonlat_to_tile(MAX_X_COORD, MAX_Y_COORD, 0.01)