import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1단계에서 저장된 area1_filtered_data.csv 파일을 불러옵니다.
try:
    area1_df = pd.read_csv('area1_filtered_data.csv')
except FileNotFoundError:
    print("Error: 'area1_filtered_data.csv' not found. Please run Stage 1 first.")
    exit()

# 지도의 최대 x, y 좌표를 찾습니다.
max_x = area1_df['x'].max()
max_y = area1_df['y'].max()

# 지도를 그립니다.
fig, ax = plt.subplots(figsize=(max_x + 1, max_y + 1)) # 지도 크기 조정

# 그리드 라인 그리기
ax.set_xticks(range(1, max_x + 2))
ax.set_yticks(range(1, max_y + 2))
ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)

# 좌측 상단이 (1,1), 우측 하단이 가장 큰 좌표가 되도록 설정
ax.set_xlim(0.5, max_x + 0.5) # 좌표가 정중앙에 오도록 0.5씩 여유
ax.set_ylim(max_y + 0.5, 0.5) # y축을 뒤집어서 (1,1)이 좌측 상단에 오도록

# 구조물 그리기
# 건설 현장을 먼저 그려서 겹칠 때 건설 현장이 위에 오도록 합니다.
for index, row in area1_df.iterrows():
    x, y = row['x'], row['y']
    struct_name = row['struct_name']

    if struct_name == '건설 현장':
        # 건설 현장은 회색 사각형, 살짝 겹치도록 크기 조절
        rect = plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, color='gray', alpha=0.7)
        ax.add_patch(rect)
    elif struct_name == '아파트' or struct_name == '빌딩':
        # 아파트와 빌딩은 갈색 원형
        circle = plt.Circle((x, y), 0.3, color='sienna', alpha=0.7)
        ax.add_patch(circle)
    elif struct_name == '반달곰 커피':
        # 반달곰 커피점은 녹색 사각형
        rect = plt.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, color='green', alpha=0.8)
        ax.add_patch(rect)
    elif struct_name == '내 집':
        # 내 집은 녹색 삼각형
        # 삼각형은 좌표를 지정해야 합니다. (x,y)를 중심으로 하는 삼각형
        triangle = plt.Polygon([(x, y - 0.3), (x - 0.3, y + 0.3), (x + 0.3, y + 0.3)], color='green', alpha=0.8)
        ax.add_patch(triangle)

# (보너스) 범례 추가
legend_patches = [
    mpatches.Rectangle((0, 0), 1, 1, fc='gray', label='건설 현장'),
    mpatches.Circle((0, 0), 0.3, fc='sienna', label='아파트/빌딩'),
    mpatches.Rectangle((0, 0), 1, 1, fc='green', label='반달곰 커피'),
    mpatches.Polygon([(0,0), (0,0), (0,0)], fc='green', label='내 집') # 실제 그릴 필요는 없음
]

# 범례 중 삼각형이 제대로 표시되지 않으므로, 따로 텍스트로 추가하거나 가장 유사한 패치를 사용
# 여기서는 간단히 사각형으로 대표하고, 설명을 붙이겠습니다.
legend_patches[3] = mpatches.Patch(facecolor='green', label='내 집 (녹색 삼각형)', edgecolor='none')


ax.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1.0, -0.05)) # 범례 위치 조정

ax.set_title("Area 1 지도")
ax.set_xlabel("X 좌표")
ax.set_ylabel("Y 좌표")

# 이미지로 저장
plt.savefig('map.png')
print("지도가 'map.png' 파일로 저장되었습니다.")

# 주피터 노트북 환경이 아니라면 plt.show()는 주석 처리하는 것이 좋음
# plt.show()