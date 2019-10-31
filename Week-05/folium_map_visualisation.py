
#geopandas is the state of the art geo-location mapping tool.
#folium is this simpler modeling style
#geopy is a geo location identifier
# Nominatim is a free open-source lat-long from strings
import folium
import requests as re
SECRET_KEY = "32e3fea63b56e30a83a74e87ee076da6"


GERMAN_STATES = ['Baden-Württemberg',
                 'Bavaria',
                 'Berlin',
                 'Brandenburg',
                 'Bremen',
                 'Hamburg',
                 'Hesse',
                 'Lower Saxony',
                 'Mecklenburg-Vorpommern',
                 'North Rhine-Westphalia',
                 'Rhineland-Palatinate',
                 'Saarland',
                 'Saxony',
                 'Saxony-Anhalt',
                 'Schleswig-Holstein',
                 'Thuringia'
                ]


from geopy.geocoders import Nominatim
geocoder = Nominatim(user_agent='me', timeout=5)

temp_data = []

for i in GERMAN_STATES:
    g = geocoder.geocode(i)
    path = f'https://api.darksky.net/forecast/{SECRET_KEY}/{g.latitude},{g.longitude}' # ,time
    response = re.get(path)
    data = response.json()
    temp = round((data['currently']['temperature'] -32) * 5/9, 2)
    temp_data.append((i,g.latitude,g.longitude,temp))


# %% mapping

germany_map = folium.Map(location=(50.6118537,9.1909725), zoom_start = 6, titles = 'cartodbpositron')

for state, lat, lon, temp in temp_data: #tuple unpacking
    custom_icon = folium.DivIcon(html=f'<div> {temp} ,</div>')
    marker = folium.Marker((lat,lon), popup=None, tooltip=(state, temp), icon=custom_icon, draggable=False)
    circle_market = folium.CircleMarket((lat, lon), radius = 20)

    marker.add_to(germany_map)
    circle_marker.add_to(germany_map)

# %% save map
germany_map.save('germany_map.html')
