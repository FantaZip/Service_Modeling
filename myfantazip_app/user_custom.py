
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import requests
import streamlit as st
from openai import OpenAI
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
from PIL import Image
import json 

Image.MAX_IMAGE_PIXELS = None 
plt.rcParams['font.family'] = 'Malgun Gothic'
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
metro = pd.read_csv("./data/metro_station_final.csv")
df = pd.read_csv("./data/total_score_final.csv")
center_df = pd.read_csv("./data/seoul_town_name_ceneter_point.csv")
rent_price_df = pd.read_csv("./data/rent_price_전세.csv")
with open('./data/bjd_region_to_codinate.geojson', 'r') as f:
    geojson_data = json.load(f)

# 권역별 자치구 분류
seoul_region = {
    "도심권(중구,종로,용산)": ["중구", "종로구","용산구"],
    "동북권(성동,동대문,강북 등)": ["성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "노원구","도봉구"],
    "서북권(은평,서대문,마포)": ["은평구", "서대문구", "마포구"],
    "동남권(강남,송파,강동 등)": ["강남구", "서초구", "송파구", "강동구"],
    "서남권(강서,구로,영등포 등)": ["양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구"],
    "전체": ["중구", "종로구","용산구","성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "노원구","도봉구",
            "은평구", "서대문구", "마포구","강남구", "서초구", "송파구", "강동구","양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구"]
}


def create_summary_df(data_frame):


    summary_df = pd.DataFrame()
    summary_df['town_name'] = data_frame['town_name']

    # 각 카테고리별 점수 계산
    summary_df['편의성'] = data_frame[['mall_score', 'mart_score', 'pharmacy_score', 'restaurant_score']].sum(axis=1) /4 *10
    summary_df['문화여가성'] = data_frame[['culture_score', 'library_score', 'cinema_score', 'park_score', 'walk_score']].sum(axis=1) /5 *10
    summary_df['교통성'] = data_frame[['bus_score', 'metro_score', 'scooter_score', 'bicycle_score']].sum(axis=1) /4 *10
    summary_df['생활 치안'] = data_frame[['cctv_score', 'light_score', 'police_score', 'crime_score']].sum(axis=1) /4 *10


    return summary_df

# 사용자 선택 함수
def search_region(region):
    if region in seoul_region:
        return seoul_region[region]
    else:
        return "선택하신 권역이 존재하지 않습니다."

def requests_chat_completion(prompt):
  response = openai_client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
      {"role":"system","content":"당신은 20~30대 사회초년생을 위한 살기 좋은 동네를 추천해주는 AI 중개인 판타입니다."},
      {"role":"user","content":prompt}
    ],
    stream=True
  )
  return response

def draw_streaming_response(response):
  st.subheader("AI 중개인 판타의 추천")
  placeholder = st.empty()
  message = ""
  for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
      message +=delta.content
      placeholder.markdown(message +  "▌")
  placeholder.markdown(message)
  

def draw_radar_chart(items, index=0):
    index_name = items.index[index]
    labels = items.columns.values[:-1]
    scores = items.iloc[index].values[:-1].round(2)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores = np.concatenate((scores, [scores[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='red', alpha=0.25)
    ax.plot(angles, scores, color='red', linewidth=2) 
    ax.set_xticklabels([]) 
    label_padding = 1.5 
    score_padding = 1.15  
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, label_padding * max(scores), label, horizontalalignment='center', verticalalignment='center', fontsize=20, color='blue')
    for angle, score in zip(angles[:-1], scores[:-1]):
        ax.text(angle, score_padding * max(scores), str(score), horizontalalignment='center', verticalalignment='center', fontsize=18, color='black')
    plt.text(0.5, 0.5, index_name, size=20, ha='center', va='center', transform=ax.transAxes)
    ax.set_aspect('equal')
    plt.show()
    return fig


def create_map(center_df, selected_town_name):
    selected_town_name = items.iloc[0].name
    town_center = center_df[center_df["emd_nm"] == selected_town_name]
    if not town_center.empty:
        center_lat = town_center["center_lati"].values[0]
        center_long = town_center["center_long"].values[0]
    else:
        center_lat, center_long = 37.5665, 126.9780 

    m = folium.Map(location=[center_lat, center_long], zoom_start=15)
    folium.Marker([center_lat, center_long], tooltip=selected_town_name).add_to(m)

    return m


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  
    return c * r

def plot_rent_info(town_name, df):
    sns.set(style="whitegrid", palette="pastel")  # 스타일 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    filtered_data = rent_price_df[rent_price_df['town_name'] == town_name]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=filtered_data, x='건물용도', y='평당평균보증금', ax=ax, errorbar=None)
    ax.set_title(f'{town_name} 전세 정보', fontsize=18, fontweight='bold')
    ax.set_ylabel('평당 평균 보증금 (단위: 만원)', fontsize=14)
    ax.set_xlabel('건물용도', fontsize=14)
    plt.xticks(rotation=45)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10),  # 텍스트 위치 조정
                    textcoords='offset points',
                    fontsize=12)
    
    return fig 

def generate_prompt(items):
    item_text=""
    weights_text = ", ".join([f"{key}:{value:.2f}" for key, value in initial_weights.items()])
    for j in range(len(items)):
      item_text += f"""
      추천 결과 {j+1}
      동네: {items.iloc[j].name}
      편의성: {items.iloc[j][0]}
      문화여가성: {items.iloc[j][1]}
      교통성: {items.iloc[j][2]}
      생활 치안: {items.iloc[j][3]}
      종합 점수: {items.iloc[j][4]}
      
      """
    #만약 추천할 동네가 상업밀집구역에 위치하면 다른 동네를 추천해주세요.  
    item_text = item_text.strip()
    prompt = f"""유저가 입력한 살기 좋은 동네의 각 지표의 선호도에 따른 추천 결과가 주어집니다.
    유저의 입력과 각 추천 결과 동네, 편의성, 문화여가성,교통성,생활 치안,종합 점수 등을 참고하여 추천 동네를 작성하세요.
    추천동네가 없다면 가장 비슷한 구역의 동네를 추천하세요.
    그 동네에 대한 정보를 검색해서 구체적으로 작성하세요.
    주변 대형마트와 백화점,영화관 정보에 대해서도 알려주세요.
    20~30대 사회초년생을 위해서 작성하세요.
    당신에 대한 소개를 먼저 하고, 친절한 말투로 작성해주세요.
    중간 중간 이모지를 적절히 사용해주세요.
    선호도가 가장 높은 지표의 정보를 검색해서 알려주세요(예시: 문화여가성->주변 카페 추천)
    사용자가 입력한 가중치 정보: {weights_text}

  ---
  유저 입력: 
  {item_text}
  ---
  """.strip()
    return prompt


st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        max-width: 800px;
        padding-top: 5rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 세부 항목과 가중치 할당
detail_items = {
    '편의성': ['쇼핑몰(백화점)', '마트&슈퍼', '약국', '음식점'],
    '문화여가성': ['문화시설(박물관&미술관)', '도서관', '영화시설', '공원', '산책로'],
    '교통성': ['버스정류장', '지하철역', '킥보드', '자전거 대여소'],
    '생활 치안': ['CCTV', '보안등', '경찰서', '범죄율']
}
item_to_column = {
    '쇼핑몰(백화점)': 'mall_score',
    '마트&슈퍼': 'mart_score',
    '약국': 'pharmacy_score',
    '음식점': 'restaurant_score',
    '문화시설(박물관&미술관)': 'culture_score',
    '도서관': 'library_score',
    '영화시설': 'cinema_score',
    '공원': 'park_score',
    '산책로': 'walk_score',
    '버스정류장': 'bus_score',
    '지하철역': 'metro_score',
    '킥보드': 'scooter_score',
    '자전거 대여소': 'bicycle_score',
    'CCTV': 'cctv_score',
    '보안등': 'light_score',
    '경찰서': 'police_score',
    '범죄율': 'crime_score'
}
# 초기 가중치 설정
initial_weights = {'교통성': 25, '문화여가성': 25, '편의성': 25, '생활 치안': 25}

# Streamlit UI 구성

col1, col2 = st.columns([3, 1])
with col2:
    st.image("./images/image_logo.png")
with col1:
    st.title('🛠️사용자 조절 도구')
selected_region = st.selectbox('원하시는 권역을 선택하세요:', list(seoul_region.keys()))

# 각 지표별로 세부 항목 선택 및 가중치 조정
col3, col4 = st.columns([1, 1])


with col3:
    # 지표별 가중치 조정
    st.subheader("지표별 가중치 조정")
    for category in initial_weights.keys():
        initial_weights[category] = st.slider(f"{category} 가중치:", 0, 100, initial_weights[category], 5, key=f"{category}_weight")
with col4:
    st.subheader("세부 항목 가중치 조정")
    for category, items in detail_items.items():
        selected_items = st.multiselect(f"선택하세요 ({category}):", options=items, key=f"{category}_items")
        st.write("")
        item_weights = {}
        if selected_items:
            for item in selected_items:
                weight = st.slider(f"{item} 가중치:", 0, 100, 100,5, key=f"{item}_weight")
                item_weights[item] = weight

        # 세부 항목 가중치 업데이트
        for item, weight in item_weights.items():
            score_col = item_to_column[item]  # 한국어 항목을 영어 열 이름으로 매핑
            if score_col in df.columns:
                df[score_col] *= (weight / 100)

# 종합점수 계산 및 상위 동네 표시
new_df = create_summary_df(df)
for category in initial_weights:
    new_df[category] *= initial_weights[category] / 100

new_df["구"] = df["county_name"]
new_df['종합점수'] = new_df[list(initial_weights.keys())].sum(axis=1) / 4
new_df.set_index('town_name', inplace=True)
new_df = new_df.round(2)
selected_gu = seoul_region[selected_region]
filtered_df = new_df[new_df['구'].isin(selected_gu)]
raw_df =st.toggle(label="raw 데이터 보기")
toggle = st.toggle(label="데이터 보기")

if raw_df:
    st.write(df)
    
if toggle:
    st.write(filtered_df)
    
items= filtered_df[['편의성', '문화여가성', '교통성', '생활 치안','종합점수']].nlargest(5, '종합점수',keep='all')
geo_score_df = new_df
geo_score_df = geo_score_df["종합점수"]
top_socre_toggle = st.toggle(label="TOP_5 보기")
if top_socre_toggle:
    st.write(items)


with st.form("form"):
    submitted = st.form_submit_button("제출")
    if submitted:
        col5, col6 = st.columns([1, 1])
        with col5:
            radar_fig = draw_radar_chart(items, index=0)
            st.pyplot(radar_fig)
        with col6:
            rent_fig = plot_rent_info(items.index[0], rent_price_df)
            st.pyplot(rent_fig)

        # 추천사 및 지도 표시
        with st.spinner("판타가 추천사를 작성합니다..."):
            prompt = generate_prompt(items)
            response = requests_chat_completion(prompt)
            draw_streaming_response(response)

        selected_town_name = items.index[0]
        st.subheader("지도(종합점수 한눈에 보기)")
        m = create_map(center_df, selected_town_name)
        folium.Choropleth(
        geo_data=geojson_data,
        data=new_df["종합점수"],
        columns=[new_df.index,new_df["종합점수"]],
        fill_color='YlOrRd',
        fill_opacity=0.5,
        line_opacity=0.3,
        key_on='feature.properties.EMD_NM').add_to(m)
        st_folium(m, width=700, height=500)
        

