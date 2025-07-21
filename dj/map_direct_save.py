import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# 1단계에서 저장된 area1_filtered_data.csv 파일을 불러옵니다.
try:
    area1_df = pd.read_csv('area1_filtered_data.csv')
except FileNotFoundError:
    print("Error: 'area1_filtered_data.csv' not found. Please run Stage 1 first.")
    exit()

# 내 집과 반달곰 커피 위치를 찾습니다.
home_location = area1_df[area1_df['struct_name'] == '내 집'].iloc[0]
cafe_locations = area1_df[area1_df['struct_name'] == '반달곰 커피']

# 최단 경로를 찾기 위한 준비
start_node = (home_location['x'], home_location['y'])
# 여러 반달곰 커피 중 가장 가까운 곳을 찾기 위해 리스트로 관리
target_nodes = [(row['x'], row['y']) for index, row in cafe_locations.iterrows()]

# 지도의 최대 x, y 좌표를 찾습니다.
max_x = int(area1_df['x'].max())
max_y = int(area1_df['y'].max())

# 그리드 맵 생성 (통과 가능 여부 판단)
# 0: 통과 가능, 1: 통과 불가능 (건설 현장)
grid_map = [[0 for _ in range(max_x + 1)] for _ in range(max_y + 1)]

# 건설 현장 위치를 맵에 표시 (통과 불가능)
for index, row in area1_df[area1_df['struct_name'] == '건설 현장'].iterrows():
    grid_map[int(row['y'])][int(row['x'])] = 1 # (y, x) 순서

# BFS 알고리즘 구현
def bfs_shortest_path(start, targets, grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])]) # (현재 위치, 경로)
    visited = set([start])

    while queue:
        (r, c), path = queue.popleft()

        # 목표 지점에 도달했는지 확인
        if (r, c) in targets:
            return path

        # 상하좌우 이동 (대각선 이동은 고려하지 않음)
        # 델타 값은 (dy, dx)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # 맵 범위 내에 있고, 방문하지 않았고, 건설 현장이 아닌 경우
            if 1 <= nr <= max_y and 1 <= nc <= max_x and \
               (nr, nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None # 경로를 찾지 못한 경우

# 최단 경로 찾기 (가장 가까운 반달곰 커피를 찾기 위해 모든 타겟에 대해 BFS 실행)
shortest_path = None
min_path_len = float('inf')

# 여러 반달곰 커피 지점 중 가장 가까운 곳을 찾기 위해
# 실제 BFS는 한 번만 돌리고, 목표가 여러 개인 경우 먼저 도달한 경로가 최단 경로가 됩니다.
# 여기서는 목표 지점 리스트를 넘겨주면 됩니다.
path_found = bfs_shortest_path(start_node, target_nodes, grid_map)

if path_found:
    shortest_path = path_found
    print("\n최단 경로가 발견되었습니다:")
    print(shortest_path)

    # 경로를 CSV 파일로 저장
    path_df = pd.DataFrame(shortest_path, columns=['x', 'y'])
    path_df.to_csv('home_to_cafe.csv', index=False)
    print("최단 경로가 'home_to_cafe.csv' 파일로 저장되었습니다.")
else:
    print("\n최단 경로를 찾을 수 없습니다.")

# 지도 시각화 (map_final.png)
fig, ax = plt.subplots(figsize=(max_x + 1, max_y + 1))

# 그리드 라인 그리기
ax.set_xticks(range(1, max_x + 2))
ax.set_yticks(range(1, max_y + 2))
ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)

# 좌측 상단이 (1,1), 우측 하단이 가장 큰 좌표가 되도록 설정
ax.set_xlim(0.5, max_x + 0.5)
ax.set_ylim(max_y + 0.5, 0.5)

# 기존 구조물 그리기 (map_draw.py와 동일)
for index, row in area1_df.iterrows():
    x, y = row['x'], row['y']
    struct_name = row['struct_name']

    if struct_name == '건설 현장':
        rect = plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, color='gray', alpha=0.7)
        ax.add_patch(rect)
    elif struct_name == '아파트' or struct_name == '빌딩':
        circle = plt.Circle((x, y), 0.3, color='sienna', alpha=0.7)
        ax.add_patch(circle)
    elif struct_name == '반달곰 커피':
        rect = plt.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, color='green', alpha=0.8)
        ax.add_patch(rect)
    elif struct_name == '내 집':
        triangle = plt.Polygon([(x, y - 0.3), (x - 0.3, y + 0.3), (x + 0.3, y + 0.3)], color='green', alpha=0.8)
        ax.add_patch(triangle)

# 최단 경로 시각화 (빨간 선)
if shortest_path:
    path_x = [p[0] for p in shortest_path]
    path_y = [p[1] for p in shortest_path]
    ax.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=4)

# (보너스) 범례 추가 (map_draw.py와 동일)
legend_patches = [
    mpatches.Rectangle((0, 0), 1, 1, fc='gray', label='건설 현장'),
    mpatches.Circle((0, 0), 0.3, fc='sienna', label='아파트/빌딩'),
    mpatches.Rectangle((0, 0), 1, 1, fc='green', label='반달곰 커피'),
    mpatches.Patch(facecolor='green', label='내 집 (녹색 삼각형)', edgecolor='none'),
    mpatches.Patch(facecolor='red', label='최단 경로', edgecolor='none')
]
ax.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1.0, -0.05))

ax.set_title("Area 1 지도 및 최단 경로")
ax.set_xlabel("X 좌표")
ax.set_ylabel("Y 좌표")

# 최종 이미지로 저장
plt.savefig('map_final.png')
print("최종 지도가 'map_final.png' 파일로 저장되었습니다.")

# plt.show()