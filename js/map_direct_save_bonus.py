import pandas as pd
import matplotlib.pyplot as plt

# --- 1단계: merged.csv 불러오기 및 전처리 ---
merged_df = pd.read_csv('merged.csv')

# NaN을 'Empty'로 변환 및 공백 제거
merged_df['struct'] = merged_df['struct'].fillna('Empty')
merged_df['struct'] = merged_df['struct'].astype(str).str.strip()

# final_type 열 생성
def get_cell_type(row):
    if row['ConstructionSite'] == 1:
        return 'ConstructionSite'
    elif row['struct'] == 'Apartment':
        return 'Apartment'
    elif row['struct'] == 'Building':
        return 'Building'
    elif row['struct'] == 'MyHome':
        return 'MyHome'
    elif row['struct'] == 'BandalgomCoffee':
        return 'BandalgomCoffee'
    else:
        return 'Empty'

merged_df['final_type'] = merged_df.apply(get_cell_type, axis=1)


# --- 2단계: 지도 시각화 함수 ---
def draw_map(df, file_name, path=None, start_node=None, end_node=None, show_legend=True):
    '''
    주어진 데이터프레임을 기반으로 지도를 시각화하여 이미지 파일로 저장합니다.

    Args:
        df (pandas.DataFrame): 지도에 표시할 데이터 (x, y, final_type 포함)
        file_name (str): 저장할 이미지 파일 이름 (예: 'map.png', 'map_final.png')
        path (list): (x, y) 튜플의 경로 리스트 (경로 시각화 시)
        start_node (tuple): 시작 노드의 (x, y) 좌표 (지도에 표시되지 않음)
        end_node (tuple): 끝 노드의 (x, y) 좌표 (지도에 표시되지 않음)
        show_legend (bool): 범례를 표시할지 여부
    '''
    max_x = df['x'].max()
    max_y = df['y'].max()

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 그리드 라인 설정 (numpy.arange 대신 list comprehension 사용)
    ax.set_xticks([x + 0.5 for x in range(max_x + 1)], minor=False)
    ax.set_yticks([y + 0.5 for y in range(max_y + 1)], minor=False)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)

    # X축 눈금을 맵 위에 그리기
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # 각 지점 플로팅
    # 범례 중복을 피하기 위해 사용
    unique_labels = {}
    for _, row in df.iterrows():
        x, y = row['x'], row['y']
        cell_type = row['final_type']

        if cell_type in ['Apartment', 'Building']:
            label = 'Apartment/Building'
            color, marker = 'saddlebrown', 'o'
        elif cell_type == 'BandalgomCoffee':
            label, color, marker = 'Bandalgom Coffee', 'green', 's'
        elif cell_type == 'MyHome':
            label, color, marker = 'My Home', 'green', '^'
        elif cell_type == 'ConstructionSite':
            label, color, marker = 'Construction Site', 'gray', 's'
            # 건설 현장은 바로 옆 좌표와 살짝 겹쳐도 되므로, 마커 크기를 약간 크게 설정
        else:
            continue

        if label not in unique_labels:
            plt.plot(x, y, marker, color=color, markersize=20, label=label)
            unique_labels[label] = True
        else:
            plt.plot(x, y, marker, color=color, markersize=20)

    # 경로 플로팅
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=10, label='Shortest Path')

    plt.title('Area Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # X, Y 축 범위 설정 및 Y축 반전 ((1,1)이 좌측 상단이 되도록)
    plt.xlim(0.5, max_x + 0.5)
    plt.ylim(max_y + 0.5, 0.5)

    # 범례 표시 (지도 오른쪽 아래)
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.7, edgecolor='black', linewidth=0.5, label='Construction Site'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='saddlebrown', markersize=12, markeredgecolor='black', markeredgewidth=0.5, label='Apartment / Building'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkgreen', markersize=12, markeredgecolor='black', markeredgewidth=0.5, label='Bandalgom Coffee'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='limegreen', markersize=14, markeredgecolor='black', markeredgewidth=0.5, label='My Home')
    ]
    if path:
        legend_items.append(plt.Line2D([0], [0], color='red', linewidth=3, alpha=0.8, label='Shortest Path'))

    ax.legend(handles=legend_items, loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    plt.xticks(list(range(1, max_x + 1)))
    plt.yticks(list(range(1, max_y + 1)))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file_name)
    plt.show()


# --- 3단계: A* 경로 탐색 및 최적 경로 탐색 ---
def _heuristic(a, b):
    '''
    A* 알고리즘의 휴리스틱 함수 (맨해튼 거리).
    Args:
        a (tuple): 시작 노드의 (x, y) 좌표
        b (tuple): 목표 노드의 (x, y) 좌표
    Returns:
        int: 맨해튼 거리
    '''
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _a_star_search(grid_width, grid_height, start, goal, impassable_cells):
    '''
    A* 알고리즘을 사용하여 그리드에서 시작점에서 목표점까지의 최단 경로를 찾습니다.
    heapq 대신 리스트 정렬을 사용합니다 (효율성은 떨어질 수 있음).
    Args:
        grid_width (int): 그리드의 최대 X 좌표
        grid_height (int): 그리드의 최대 Y 좌표
        start (tuple): 시작 노드의 (x, y) 좌표 (1-인덱스)
        goal (tuple): 목표 노드의 (x, y) 좌표 (1-인덱스)
        impassable_cells (set): 통과할 수 없는 (x, y) 튜플 집합
    Returns:
        list: (x, y) 튜플로 이루어진 최단 경로 리스트, 경로가 없으면 None
    '''
    if start in impassable_cells or goal in impassable_cells:
        return None # 시작점 또는 목표점이 통과 불가능한 지점인 경우
    
    # heapq 대신 일반 리스트와 sort()를 사용하여 우선순위 큐 구현
    frontier = [(0, start)]
    came_from = {}
    g_cost = {start: 0}
    f_cost = {start: _heuristic(start, goal)}

    while frontier:
        # 가장 작은 f_cost를 가진 노드를 추출
        frontier.sort()
        _, current = frontier.pop(0)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] # 경로를 시작점에서 목표점 순서로 뒤집기

        # 가능한 이동 (상, 하, 좌, 우)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # 이웃이 그리드 범위 내에 있는지 확인
            if not (1 <= neighbor[0] <= grid_width and 1 <= neighbor[1] <= grid_height):
                continue
            
            # 이웃이 통과 불가능한 지점인지 확인
            if neighbor in impassable_cells:
                continue
            
            new_g = g_cost[current] + 1 # 이웃으로 이동하는 비용은 1
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                f_cost[neighbor] = new_g + _heuristic(neighbor, goal)
                
                # 새로운 노드를 frontier에 추가하고 정렬
                frontier.append((f_cost[neighbor], neighbor))
                frontier.sort()
                came_from[neighbor] = current
    return None #경로 찾을 수 없음

def _generate_permutations(elements):
    '''
    itertools.permutations를 대체하여 리스트의 모든 순열을 재귀적으로 생성합니다.
    Args:
        elements (list): 순열을 생성할 요소들의 리스트
    Returns:
        list: 요소들의 모든 가능한 순열을 담은 리스트의 리스트
    '''
    if len(elements) == 0:
        return [[]]
    if len(elements) == 1:
        return [elements]
    perms = []
    for i in range(len(elements)):
        m = elements[i]
        rem = elements[:i] + elements[i+1:]
        for p in _generate_permutations(rem):
            perms.append([m] + p)
    return perms

def find_optimal_path_visiting_all_structures(grid_width, grid_height, start, waypoints, impassable_cells):
    '''
    지정된 모든 구조물을 방문하는 최적의 경로를 찾습니다.
    세그먼트에는 A*를 사용하고, TSP 부분에는 _generate_permutations를 사용합니다.
    Args:
        grid_width (int): 그리드의 최대 X 좌표
        grid_height (int): 그리드의 최대 Y 좌표
        start (tuple): 시작 노드의 (x, y) 좌표 (1-인덱스)
        end (tuple): 목표 노드의 (x, y) 좌표 (1-인덱스)
        structures_to_visit (list): 방문해야 할 모든 구조물 (x, y) 튜플 리스트
        impassable_cells (set): 통과할 수 없는 (x, y) 튜플 집합
    Returns:
        list: (x, y) 튜플로 이루어진 최적의 전체 경로 리스트, 경로가 없으면 None
    '''
    all_nodes = [start] + waypoints
    best_path = None
    min_len = float('inf')
    for perm in _generate_permutations(waypoints):
        sequence = [start] + list(perm)
        total_len = 0
        full_path = []
        valid = True
        for i in range(len(sequence) - 1):
            seg = _a_star_search(grid_width, grid_height, sequence[i], sequence[i+1], impassable_cells)
            if seg is None:
                valid = False
                break
            full_path.extend(seg if i == 0 else seg[1:])
            total_len += len(seg) - 1
        if valid and total_len < min_len:
            best_path = full_path
            min_len = total_len
    return best_path


# --- 실행 메인 ---
if __name__ == '__main__':
    max_x = merged_df['x'].max()
    max_y = merged_df['y'].max()

    impassable = set(tuple(xy) for xy in merged_df[merged_df['final_type'] == 'ConstructionSite'][['x', 'y']].values)
    #draw_map(merged_df, 'map.png')

    my_home_candidates = merged_df[merged_df['final_type'] == 'MyHome'][['x', 'y']]
    cafe_candidates = merged_df[merged_df['final_type'] == 'BandalgomCoffee'][['x', 'y']]

    if my_home_candidates.empty:
        print("Error: MyHome not found on the map.")
        exit()
    if cafe_candidates.empty:
        print("Error: Bandalgom Coffee not found on the map.")
        exit()

    my_home = tuple(my_home_candidates.values[0])
    cafes = [tuple(xy) for xy in cafe_candidates.values]

    # 중간 방문지에는 건물들과 모든 카페 포함
    waypoints = [tuple(xy) for xy in merged_df[merged_df['final_type'].isin(['Apartment', 'Building', 'BandalgomCoffee'])][['x', 'y']].values if tuple(xy) not in impassable]

    path = find_optimal_path_visiting_all_structures(max_x, max_y, my_home, waypoints, impassable)

    if path:
        pd.DataFrame(path, columns=['x', 'y']).to_csv('home_to_cafe.csv', index=False)
        draw_map(merged_df, 'map_final_bonus.png', path=path, start_node=my_home, end_node=path[-1])
    else:
        draw_map(merged_df, 'map_final_bonus.png')