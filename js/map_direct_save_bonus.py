import pandas as pd
import matplotlib.pyplot as plt

# 리스트 기반 BFS로 경로 찾기 (탐색 깊이 제한)
def bfs_path(start, goal, is_blocked, width, height, max_depth = 100):
    queue = [(start, [start], 0)]
    visited = set()
    visited.add(start)

    while queue:
        current, path, depth = queue.pop(0)
        if current == goal:
            return path
        if depth >= max_depth:
            continue

        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx <= width and 0 <= ny <= height:
                if (nx, ny) not in visited and not is_blocked(ny, nx):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)], depth + 1))
    return None

# 데이터 로드
merged = pd.read_csv('merged.csv')
merged['struct'] = merged['struct'].apply(lambda x: x.strip() if isinstance(x, str) else x)

# 좌표 -> 공사장 여부 사전 생성 (공사장이면 무조건 막힘)
construction_coords = set(
    tuple(xy) for xy in merged[merged['ConstructionSite'] == 1][['x', 'y']].values.tolist()
)

def is_blocked(y, x):
    return (x, y) in construction_coords

# 시작점, 도착점, 경유지
start = tuple(merged[merged['struct'] == 'MyHome'][['x','y']].values[0])
goals = list(merged[merged['struct'] == 'BandalgomCoffee'][['x','y']].itertuples(index=False, name=None))

# 건물 위치 (공사장이 아닌 경우만)
waypoints = list(
    merged[
        (merged['struct'].isin(['Apartment', 'Building'])) &
        (merged['ConstructionSite'] == 0)
    ][['x','y']].itertuples(index = False, name = None)
)

# 모든 경유지 + 모든 도착지 순회
all_targets = waypoints + goals

# 휴리스틱 기반 경로 생성 (가까운 곳부터 순차 방문)
def find_path_through_all_waypoints(start, targets):
    remaining = targets.copy()
    current = start
    path = []

    while remaining:
        best_segment = None
        best_pt = None

        for pt in remaining:
            segment = bfs_path(current, pt, is_blocked, merged.x.max(), merged.y.max(), max_depth = 150)
            if segment and (best_segment is None or len(segment) < len(best_segment)):
                best_segment = segment
                best_pt = pt

        if best_segment is None:
            raise Exception(f'No path to remaining waypoint from {current}')

        path += best_segment[:-1]
        current = best_pt
        remaining.remove(best_pt)

    path.append(current)
    return path

best_path = find_path_through_all_waypoints(start, all_targets)

# CSV 저장
pd.DataFrame(best_path, columns = ['x', 'y']).to_csv('home_to_cafe_with_waypoints.csv', index = False)

# 시각화
plt.figure(figsize = (10, 10))
plot_data = merged[~(merged['struct'].isna() & (merged['ConstructionSite'] == 0))]
symbols = {
    'Apartment': {'marker': 'o', 'color': 'brown', 'label': 'Apartment'},
    'Building': {'marker': 'o', 'color': 'brown', 'label': 'Building'},
    'BandalgomCoffee': {'marker': 's', 'color': 'green', 'label': 'Bandalgom Coffee'},
    'MyHome': {'marker': '^', 'color': 'green', 'label': 'My Home'},
    'ConstructionSite': {'marker': 's', 'color': 'gray', 'label': 'Construction Site'}
}

construction = plot_data[plot_data['ConstructionSite'] == 1]
plt.scatter(construction['x'], construction['y'], s = 2000, marker = 's', color = 'gray', label = 'Construction')

for struct_type in ['Apartment', 'Building', 'BandalgomCoffee', 'MyHome']:
    data = plot_data[(plot_data['struct'] == struct_type) & (plot_data['ConstructionSite'] == 0)]
    plt.scatter(data['x'], data['y'],
                marker = symbols[struct_type]['marker'],
                color = symbols[struct_type]['color'],
                label = symbols[struct_type]['label'],
                s = 500)

path_x = [p[0] for p in best_path]
path_y = [p[1] for p in best_path]
plt.plot(path_x, path_y, color = 'red', linewidth = 3, label = 'Heuristic Path')

plt.gca().invert_yaxis()
plt.grid(True)
plt.xticks(range(merged.x.min(), merged.x.max() + 1), labels = [''] * (merged.x.max() - merged.x.min() + 1))
plt.yticks(range(merged.y.min(), merged.y.max() + 1))

for x in range(merged.x.min(), merged.x.max() + 1):
    plt.text(x, merged.y.min() - 0.1, str(x), ha = 'center', va = 'bottom', fontsize = 10)

plt.title('Heuristic Path Visiting All Structures and Cafes', pad = 30)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), ncol = 2, markerscale = 0.3)

plt.tight_layout()
plt.savefig('map_with_waypoints_heuristic.png')
plt.show()
