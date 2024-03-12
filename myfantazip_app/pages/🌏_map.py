import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from math import radians, cos, sin, asin, sqrt




subway_stations = pd.read_csv('./data/metro_station_final.csv')
pharmacies = pd.read_csv('./data/pharmacy.csv')
bus_stops = pd.read_csv('./data/seoul_bus_stop.csv')
market = pd.read_csv('./data/mart_and_market.csv')
department_store = pd.read_csv("./data/department_store.csv")
shopping_mall = pd.read_csv("./data/shopping_mall.csv")
center_point = pd.read_csv('./data/seoul_town_name_ceneter_point.csv')
park = pd.read_csv('./data/park.csv')


st.title('동네 기반 시설 지도 서비스')


town_name = st.text_input('동 이름을 입력하세요:')
col1, col2 = st.columns(2)
with col1:
    show_subway = st.checkbox('지하철역')
    show_pharmacies = st.checkbox('약국')
    show_bus_stops = st.checkbox('버스정류장')
    show_market = st.checkbox('대형마트&슈퍼')
            
with col2:
    show_park = st.checkbox('공원')
    show_department_store = st.checkbox('백화점')
    show_shopping_mall = st.checkbox('쇼핑몰')



#반경 설정
radius = st.slider('반경을 설정하세요 (km):', min_value=0.1, max_value=5.0, value=1.0, step=0.1)

def town_center_point(town_name, center_point):
    town_center = center_point[center_point['emd_nm'] == town_name]

    if not town_center.empty:
        center_long = town_center['center_long'].values[0]
        center_lat = town_center['center_lati'].values[0]
        return center_lat,center_long
    else:

        return 37.5665,126.9780
  
center_lat, center_long = town_center_point(town_name, center_point)
  

m = folium.Map(location=[center_lat, center_long], zoom_start=14)

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371 
    return c * r


def add_markers(dataframe, category_name,radius):
    for index, row in dataframe.iterrows():
        lat, long = row['y'], row['x']
        if haversine(center_long, center_lat, long, lat) <= radius:  # 반경 1km 이내
            folium.Marker([lat, long], popup=f"{row['name']} ({category_name})").add_to(m)

if show_subway:
    add_markers(subway_stations, '지하철역', radius)

if show_pharmacies:
    add_markers(pharmacies, '약국', radius)

if show_bus_stops:
    add_markers(bus_stops, '버스정류장', radius)

if show_market:
    add_markers(market, '대형마트&슈퍼', radius)

if show_park:
  add_markers(park,'공원', radius)
  
if show_department_store:
  add_markers(department_store,'백화점', radius)
  
if show_shopping_mall:
  add_markers(shopping_mall,'쇼핑몰', radius)

# 스트림릿에 지도 표시
folium_static(m)


