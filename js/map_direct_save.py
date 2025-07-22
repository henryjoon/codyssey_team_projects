import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv("merged.csv")
df["struct"] = df["struct"].apply(lambda x: x.strip() if isinstance(x, str) else x)

# 2. 출발점(MyHome), 도착점(BandalgomCoffee)
start = df[df["struct"] == "MyHome"][["x", "y"]].iloc[0]
start = (int(start["x"]), int(start["y"]))
goals = df[df["struct"] == "BandalgomCoffee"][["x", "y"]].apply(tuple, axis=1).tolist()

# 3. 장애물 좌표 목록 생성
obstacles = df[df["ConstructionSite"] == 1][["x", "y"]].apply(tuple, axis=1).tolist()

# 4. BFS 함수 (deque 없이 리스트로 큐 구현)
def bfs(start, goals, obstacles):
    visited = set()
    queue = [(start, [start])]  # (현재 좌표, 경로 리스트)
    visited.add(start)

    while queue:
        current, path = queue.pop(0)
        if current in goals:
            return path

        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if (0 <= nx <= df["x"].max() and
                0 <= ny <= df["y"].max() and
                neighbor not in visited and
                neighbor not in obstacles):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# 5. 최단 경로 탐색
path = bfs(start, goals, obstacles)

# 6. 경로 CSV 저장
pd.DataFrame(path, columns=["x", "y"]).to_csv("home_to_cafe.csv", index=False)

# 7. 지도 시각화
plt.figure(figsize=(10, 10))

# 공사장
construction = df[df["ConstructionSite"] == 1]
plt.scatter(construction["x"], construction["y"], s=500, marker="s", color="gray", label="Construction")

# 구조물
for struct, marker, color in [
    ("Apartment", "o", "brown"),
    ("Building", "o", "brown"),
    ("BandalgomCoffee", "s", "green"),
    ("MyHome", "^", "green")
]:
    data = df[(df["struct"] == struct) & (df["ConstructionSite"] == 0)]
    plt.scatter(data["x"], data["y"], s=500, marker=marker, color=color, label=struct)

# 경로 선
path_x = [p[0] for p in path]
path_y = [p[1] for p in path]
plt.plot(path_x, path_y, color="red", linewidth=3, label="Shortest Path")

# 눈금/격자 설정
plt.gca().invert_yaxis()
plt.grid(True)
plt.xticks(range(df["x"].min(), df["x"].max() + 1), labels=[""] * (df["x"].max() - df["x"].min() + 1))
plt.yticks(range(df["y"].min(), df["y"].max() + 1))

# 상단에 x 좌표 텍스트 표시
top_y = df["y"].min() - 0.1
for x in range(df["x"].min(), df["x"].max() + 1):
    plt.text(x, top_y, str(x), ha="center", va="bottom", fontsize=10)

plt.title("Map Visualization with BFS Path", pad=30)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, markerscale=0.3)
plt.tight_layout()
plt.savefig("map_final.png")
plt.show()
