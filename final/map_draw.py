import pandas as pd
import matplotlib.pyplot as plt

# CSV 불러오기
merged = pd.read_csv('merged.csv')

# 공백 제거, NaN 유지
merged['struct'] = merged['struct'].apply(lambda x: x.strip() if isinstance(x, str) else x)

# 시각화 대상: 건물 있거나 공사장인 셀만 필터링
plot_data = merged[~(merged['struct'].isna() & (merged['ConstructionSite'] == 0))]

# 구조물별 마커, 색상, 라벨
symbols = {
    'Apartment': {'marker': 'o', 'color': 'brown', 'label': 'Apartment / Building'},
    'Building': {'marker': 'o', 'color': 'brown', 'label': 'Apartment / Building'},
    'BandalgomCoffee': {'marker': 's', 'color': 'green', 'label': 'Bandalgom Coffee'},
    'MyHome': {'marker': '^', 'color': 'green', 'label': 'My Home'},
    'ConstructionSite': {'marker': 's', 'color': 'gray', 'label': 'Construction Site'}
}

# 지도 시각화
plt.figure(figsize = (10, 10))
ax = plt.gca()

# 그리드 설정
max_x = merged['x'].max()
max_y = merged['y'].max()
ax.set_xticks([x + 0.5 for x in range(max_x + 1)])
ax.set_yticks([y + 0.5 for y in range(max_y + 1)])
ax.grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.5)
ax.set_xticks(range(1, max_x + 1))
ax.set_yticks(range(1, max_y + 1))
ax.set_xticklabels(range(1, max_x + 1))
ax.set_yticklabels(range(1, max_y + 1))
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# 제목과 축 설정
plt.title('Map', pad = 20)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0.5, max_x + 0.5)
plt.ylim(max_y + 0.5, 0.5)
plt.gca().set_aspect('equal', adjustable = 'box')

# 공사장 먼저 시각화
construction = plot_data[plot_data['ConstructionSite'] == 1]
plt.scatter(construction['x'], construction['y'],
            marker = symbols['ConstructionSite']['marker'],
            color = symbols['ConstructionSite']['color'],
            label = symbols['ConstructionSite']['label'],
            s = 2000)

# 구조물 시각화 (공사장이 아닌 셀만)
used_labels = set()
for struct_type in ['Apartment', 'Building', 'BandalgomCoffee', 'MyHome']:
    data = plot_data[(plot_data['struct'] == struct_type) & (plot_data['ConstructionSite'] == 0)]
    label = symbols[struct_type]['label']
    
    # 중복 라벨 제거
    if label not in used_labels:
        show_label = label
        used_labels.add(label)
    else:
        show_label = None

    plt.scatter(data['x'], data['y'],
                marker = symbols[struct_type]['marker'],
                color = symbols[struct_type]['color'],
                label  =show_label,
                s = 500)

plt.legend(loc = 'lower right', fontsize = 10, markerscale = 0.3, frameon = True)

# 범례 설정
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc = 'lower right', frameon = True, fontsize = 10, markerscale = 0.3)

# 저장 및 출력
plt.tight_layout()
plt.savefig('map.png')
plt.show()
