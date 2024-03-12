
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

plt.rcParams['font.family'] = 'Malgun Gothic'
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
metro = pd.read_csv("./data/metro_station_final.csv")
df = pd.read_csv("./data/total_score_test.csv")
center_df = pd.read_csv("./data/seoul_town_name_ceneter_point.csv")
# 권역별 자치구 분류
seoul_region = {
    "도심권": ["중구", "종로구","용산구"],
    "동북권": ["성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "노원구","도봉구"],
    "서북권": ["은평구", "서대문구", "마포구"],
    "동남권": ["강남구", "서초구", "송파구", "강동구"],
    "서남권": ["양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구"],
    "전역권": ["중구", "종로구","용산구","성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "노원구","도봉구",
            "은평구", "서대문구", "마포구","강남구", "서초구", "송파구", "강동구","양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구"]
}

detail_items = {
    '편의성': ['쇼핑몰(백화점)','마트&슈퍼', '약국', '음식점'],
    '문화여가성': ['문화시설(박물관&미술관)', '도서관', '영화시설','공원','산책로'],
    '교통성': ['버스정류장', '지하철역', '킥보드', '자전거 대여소'],
    '생활 치안': ['CCTV', '보안등', '경찰서', '범죄율']
}
def create_summary_df(data_frame):
    """
    이 함수는 주어진 데이터 프레임을 바탕으로 새로운 요약 데이터 프레임을 생성합니다.
    각 동네의 '편의성', '문화여가성', '교통성', '생활 치안' 점수를 계산하고,
    이들의 합으로 '종합점수'를 계산합니다.

    Parameters:
    - data_frame (pd.DataFrame): 원본 데이터가 포함된 데이터 프레임

    Returns:
    - pd.DataFrame: 새로운 요약 데이터 프레임
    """
    # 새로운 데이터 프레임 초기화
    summary_df = pd.DataFrame()
    summary_df['town_name'] = data_frame['town_name']

    # 각 카테고리별 점수 계산
    summary_df['편의성'] = data_frame[['mall_score', 'mart_score', 'pharmacy_score', 'restaurant_score']].sum(axis=1)
    summary_df['문화여가성'] = data_frame[['culture_score', 'library_score', 'cinema_score', 'park_score', 'walk_score']].sum(axis=1)
    summary_df['교통성'] = data_frame[['bus_score', 'metro_score', 'scooter_score', 'bicycle_score']].sum(axis=1)
    summary_df['생활 치안'] = data_frame[['cctv_score', 'light_score', 'police_score', 'crime_score']].sum(axis=1)


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
    scores = np.concatenate((scores,[scores[0]]))  
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='red', alpha=0.25)
    ax.plot(angles, scores, color='red', linewidth=3)

    ax.set_xticks([])

    for angle, score in zip(angles[:-1], scores[:-1]):
        ax.text(angle, score + 5, str(score), horizontalalignment='center', verticalalignment='center', fontsize=20, color='black')
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 50, label, horizontalalignment='center', verticalalignment='center', fontsize=20, color='blue')
    plt.text(0.5, 0.1, index_name, size=14, ha='center', transform=fig.transFigure)
    return fig  
  
def create_map(center_df, selected_town_name):
    # 선택된 동네의 중심 좌표를 가져옵니다.
    selected_town_name = items.iloc[0].name
    town_center = center_df[center_df["emd_nm"] == selected_town_name]
    if not town_center.empty:
        center_lat = town_center["center_lati"].values[0]
        center_long = town_center["center_long"].values[0]
    else:
        # 기본값으로 설정합니다.
        center_lat, center_long = 37.5665, 126.9780  # 서울의 중심 좌표

    # 지도 생성
    m = folium.Map(location=[center_lat, center_long], zoom_start=15)

    # 선택된 동네에 마커 추가
    folium.Marker([center_lat, center_long], tooltip=selected_town_name).add_to(m)

    return m


# 하버사인 공식을 이용해 두 좌표 간의 거리를 계산하는 함수
def haversine(lon1, lat1, lon2, lat2):
    # 위도와 경도를 라디안으로 변환
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # 라디안 차이 계산
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 

    # 하버사인 공식
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # 지구의 반지름(km)
    return c * r


# 스타일링을 위한 CSS
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

st.title('사용자 조절 도구')
st.markdown('### 각 지표의 가중치를 조정하세요. 총합은 100이 되어야 합니다.')

  # 초기 가중치 값과 전체 가중치 설정
initial_weights = {'교통성': 25, '문화여가성': 25, '편의성': 25, '생활 치안': 25}
total_weight = 100
# 권역 선택을 위한 selectbox 추가
selected_region = st.selectbox('권역을 선택하세요:', list(seoul_region.keys()))

  # 세션 상태 초기화
if 'weights' not in st.session_state:
    st.session_state.weights = initial_weights.copy()

  # 각 지표에 대한 슬라이더
for category in initial_weights.keys():
    st.session_state.weights[category] = st.slider(category, 0, 100, st.session_state.weights[category], 5, key=category)
    selected_items = st.multiselect(
            f"{category} 세부 항목 선택:",
            options=detail_items[category],
            default=detail_items[category],
            key=f"{category}_items"
        )
    st.markdown("<br><br>", unsafe_allow_html=True)


  # 사용된 가중치와 남은 가중치 계산
used_weight = sum(st.session_state.weights.values())
remaining_weight = total_weight - used_weight

# 사용된 가중치와 남은 가중치 표시
st.markdown(f'<p class="big-font">사용된 가중치: {used_weight}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="big-font">남은 가중치: {remaining_weight}</p>', unsafe_allow_html=True)

# 경고 메시지 표시
if used_weight > total_weight:
    st.error('가중치의 총합이 100을 초과하였습니다. 다시 조정해 주세요.')
elif used_weight < total_weight:
    st.warning('가중치의 총합이 100에 미치지 않습니다. 남은 가중치를 할당해 주세요.')
else:
    st.success('설정이 완료되었습니다.')


# 스트림릿 세션 상태에서 가중치 가져오기
weights = st.session_state.weights

# 새로운 DataFrame 생성
new_df = create_summary_df(df)
# 각 지표의 가중치 곱한 값 계산 및 저장
for category in ['편의성', '문화여가성', '교통성', '생활 치안']:
    new_df[category] = new_df[category] * weights[category] / 100
new_df["구"] = df["county_name"]
# 종합점수 계산 및 저장
# 카테고리 이름을 리스트로 지정하여 해당 칼럼들만 선택
categories = ['편의성', '문화여가성', '교통성', '생활 치안']
new_df['종합점수'] = new_df[categories].sum(axis=1)
max_total_score = new_df['종합점수'].max()

# '종합점수'를 100점 만점으로 스케일링
new_df['종합점수'] = (new_df['종합점수'] / max_total_score) * 100
new_df.set_index('town_name', inplace=True)
new_df = new_df.round(2)
selected_gu = seoul_region[selected_region]
filtered_df = new_df[new_df['구'].isin(selected_gu)]
toggle = st.toggle(label="데이터 보기")



# 슬라이더 값이 1일 때만 데이터프레임 표시
if toggle:
    st.write(filtered_df)
items= filtered_df[['편의성', '문화여가성', '교통성', '생활 치안','종합점수']].nlargest(3, '종합점수',keep='all')

top_socre_toggle = st.toggle(label="TOP_3 보기")
if top_socre_toggle:
    st.write(items)


def generate_prompt(items):
    item_text=""
    weights_text = ", ".join([f"{key}:{value:.2f}" for key, value in weights.items()])
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
      
    item_text = item_text.strip()
    prompt = f"""유저가 입력한 살기 좋은 동네의 각 지표의 선호도에 따른 추천 결과가 주어집니다.
    유저의 입력과 각 추천 결과 동네, 편의성, 문화여가성,교통성,생활 치안,종합 점수 등을 참고하여 추천 동네를 작성하세요.
    만약 추천할 동네가 도심(상업밀집구역 예:을지로,여의도,종로1가 등등)에 위치하면 다른 동네를 추천해주세요 
    추천동네가 없다면 가장 비슷한 구역의 동네를 추천하세요
    그 동네에 대한 정보를 검색해서 구체적으로 작성하세요.
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
from PIL import Image

  # 이미지 최대 픽셀 수 제한 늘리기
Image.MAX_IMAGE_PIXELS = None 

with st.form("form"):
    submitted = st.form_submit_button("제출")
    if submitted:
        fig = draw_radar_chart(items, index=0)
        st.pyplot(fig)

        with st.spinner("판타가 추천사를 작성합니다..."):
            prompt = generate_prompt(items)
            response = requests_chat_completion(prompt)
            draw_streaming_response(response)

        # TOP 3 중 첫 번째 동네의 이름을 가져옵니다.
        selected_town_name = items.index[0]

        # 지도 섹션
        st.subheader("지도")
        m = create_map(center_df, selected_town_name) # 지도 생성 함수 호출
        st_folium(m, width=700, height=500)

