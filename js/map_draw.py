import pandas as pd
import matplotlib.pyplot as plt

# CSV 불러오기, merged라는 변수에 저장
merged = pd.read_csv("merged_all.csv")

# 시각화 대상: 건물 있거나 공사장인 셀만 필터링 .isna(): Nan값인지 아닌지 Bool 반환
plot_data = merged[~(merged["struct"].isna() & (merged["ConstructionSite"] == 0))]

# 구조물별 기호와 색상 정의 marker모양, color색깔, label범례에 표시될 이름
symbols = {
    "Apartment": {"marker": "o", "color": "brown", "label": "Apartment"},
    "Building": {"marker": "o", "color": "brown", "label": "Building"},
    "BandalgomCoffee": {"marker": "s", "color": "green", "label": "Bandalgom Coffee"},
    "MyHome": {"marker": "^", "color": "green", "label": "My Home"},
    "ConstructionSite": {"marker": "s", "color": "gray", "label": "ConstructionSite"}
}

# 그래프 생성: 10인치 x 10인치로 사이즈 설정
plt.figure(figsize=(10, 10))

# 구조물별 시각화
for struct_type in ["Apartment", "Building", "BandalgomCoffee", "MyHome"]: # 시각화 할 구조물 목록
    data = plot_data[plot_data["struct"] == struct_type] # 
    plt.scatter(data["x"], data["y"],
                marker=symbols[struct_type]["marker"],
                color=symbols[struct_type]["color"],
                label=symbols[struct_type]["label"],
                s=100)

# 6. ConstructionSite만 있는 셀
construction = plot_data[plot_data["struct"].isna() & (plot_data["ConstructionSite"] == 1)]
plt.scatter(construction["x"], construction["y"],
            marker=symbols["ConstructionSite"]["marker"],
            color=symbols["ConstructionSite"]["color"],
            label=symbols["ConstructionSite"]["label"],
            s=100)

# 7. y축 상하반전 및 눈금/격자 설정
plt.gca().invert_yaxis()
plt.grid(True)
plt.xticks(range(merged["x"].min(), merged["x"].max() + 1))
plt.yticks(range(merged["y"].min(), merged["y"].max() + 1))
plt.title("Map Visualization")
plt.xlabel("X")
plt.ylabel("Y")

# 8. 범례 (지도 가리지 않도록 하단 배치)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

# 9. 출력 및 저장
plt.tight_layout()
plt.savefig("map.png")
plt.show()

# 모양 사이즈 키우기
# 좌표 번호 상단에 하기