import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기 및 전처리
df = pd.read_csv('merged.csv')
df['struct'] = df['struct'].apply(lambda x: x.strip() if isinstance(x, str) else x)

# 출발점(MyHome), 도착점(BandalgomCoffee)
start = df[df['struct'] == 'MyHome'][['x', 'y']].iloc[0]
start = (int(start['x']), int(start['y']))
goals = df[df['struct'] == 'BandalgomCoffee'][['x', 'y']].apply(tuple, axis=1).tolist()

# 장애물 좌표 목록 생성
obstacles = df[df['ConstructionSite'] == 1][['x', 'y']].apply(tuple, axis=1).tolist()

# BFS 함수 (deque 없이 리스트로 큐 구현)
def bfs(start, goals, obstacles):
    visited = set()
    queue = [(start, [start])]
    visited.add(start)

    while queue:
        current, path = queue.pop(0)
        if current in goals:
            return path

        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if (0 <= nx <= df['x'].max() and
                0 <= ny <= df['y'].max() and
                neighbor not in visited and
                neighbor not in obstacles):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# 최단 경로 탐색
path = bfs(start, goals, obstacles)

# 경로 CSV 저장
pd.DataFrame(path, columns=['x', 'y']).to_csv('home_to_cafe.csv', index=False)

# 지도 시각화
plt.figure(figsize=(10, 10))
ax = plt.gca()

# 그리드 설정
max_x = df['x'].max()
max_y = df['y'].max()
ax.set_xticks([x + 0.5 for x in range(max_x + 1)])
ax.set_yticks([y + 0.5 for y in range(max_y + 1)])
ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
ax.set_xticks(range(1, max_x + 1))
ax.set_yticks(range(1, max_y + 1))
ax.set_xticklabels(range(1, max_x + 1))
ax.set_yticklabels(range(1, max_y + 1))
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# 공사장 시각화
construction = df[df['ConstructionSite'] == 1]
plt.scatter(construction['x'], construction['y'], s=2000, marker='s', color='gray', label='Construction Site')

# 구조물 시각화
for struct, marker, color, label in [
    ('Apartment', 'o', 'brown', 'Apartment / Building'),
    ('Building', 'o', 'brown', 'Apartment / Building'),
    ('BandalgomCoffee', 's', 'darkgreen', 'Bandalgom Coffee'),
    ('MyHome', '^', 'limegreen', 'My Home')
]:
    data = df[(df['struct'] == struct) & (df['ConstructionSite'] == 0)]
    if not data.empty:
        plt.scatter(data['x'], data['y'], s=500, marker=marker, color=color, label=label)

# 경로 시각화
if path:
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=6, label='Shortest Path')

# 제목과 축 설정
plt.title('Area Map', pad=20)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0.5, max_x + 0.5)
plt.ylim(max_y + 0.5, 0.5)
plt.gca().set_aspect('equal', adjustable='box')

# 범례 설정
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right', frameon=True, fontsize=10, markerscale = 0.3)

plt.tight_layout()
plt.savefig('map_final.png')
plt.show()
