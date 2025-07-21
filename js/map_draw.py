import pandas as pd
import matplotlib.pyplot as plt

# CSV 불러오기, merged라는 변수에 저장
merged = pd.read_csv("merged.csv")

# 공백 제거, NaN 유지
merged["struct"] = merged["struct"].apply(lambda x: x.strip() if isinstance(x, str) else x)


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

# 1. 공사장 먼저 시각화 (무조건 우선)
construction_all = plot_data[plot_data["ConstructionSite"] == 1]
plt.scatter(construction_all["x"], construction_all["y"],
            marker=symbols["ConstructionSite"]["marker"],
            color=symbols["ConstructionSite"]["color"],
            label=symbols["ConstructionSite"]["label"],
            s=300)

# 2. 구조물 시각화 (공사장이 아닌 경우만)
for struct_type in ["Apartment", "Building", "BandalgomCoffee", "MyHome"]:
    data = plot_data[
        (plot_data["struct"] == struct_type) &
        (plot_data["ConstructionSite"] == 0)  # 공사장 아닌 셀만 시각화
    ]
    plt.scatter(data["x"], data["y"],
                marker=symbols[struct_type]["marker"],
                color=symbols[struct_type]["color"],
                label=symbols[struct_type]["label"],
                s=300)


# 8. y축 상하 반전 및 그리드
plt.gca().invert_yaxis()
plt.grid(True)

# 9. 축 범위 및 눈금 설정
x_min, x_max = merged["x"].min(), merged["x"].max()
y_min, y_max = merged["y"].min(), merged["y"].max()
plt.xticks(ticks=range(x_min, x_max + 1), labels=[""] * (x_max - x_min + 1))
plt.yticks(range(y_min, y_max + 1))

# ✅ x좌표를 플롯 내부의 상단에 추가 (y_max보다 약간 위에 표시)
top_y = y_min - 0.1 # y축이 invert되어 있으므로 숫자가 클수록 아래임
for x in range(x_min, x_max + 1):
    plt.text(x, top_y, str(x), ha='center', va='bottom', fontsize=10,)



# 10. 제목, 레이블, 범례
plt.title("Map Visualization", pad = 30)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, markerscale = 0.5)

# 11. 출력 및 저장
plt.tight_layout()
plt.savefig("map.png")
plt.show()

# 모양 사이즈 키우기
# 좌표 번호 상단에 하기