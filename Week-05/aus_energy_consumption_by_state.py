import folium
import pandas as pd
from geopy.geocoders import Nominatim

state_geo = r'Data/aus_states.json'
energy_usage = r'Data/aus_summary_energy_usage.csv'
states = pd.read_csv(energy_usage, index_col =0)[:-1]
states
states['lat'], states['long'] = 0, 0
geocoder = Nominatim(country_bias='au', user_agent='POTUS', timeout=20)

# %% Get Geolocations of center of each State
for state in states.index:
    g = geocoder.geocode(state)
    states.at[state,'lat'] = g.latitude
    states.at[state,'long'] = g.longitude

# %%
map = folium.Map(location=[-23.6993532, 133.8713752], zoom_start=5)
folium.GeoJson(state_geo).add_to(map)

for energy, lat, long in zip(states['Renewables'], states['lat'], states['long']):
    folium.Marker((lat,long), popup=None, tooltip=(energy)).add_to(map)

# %% save map
map.save('aus_states.html')

# choro = folium.Choropleth(
#     geo_data=state_geo,
#     name='choropleth',
#     data=states,
#     columns=[states.index, 'Renewables'],
#     key_on='feature.id',
#     fill_color='YlGn',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name='Renewable Energy').add_to(map)
