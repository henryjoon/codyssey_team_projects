# map_direct_save.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def load_and_process_data(area_category_path, area_map_path, area_struct_path):

    merged_df = pd.read_csv('merged.csv')
    merged_df['struct'] = merged_df['struct'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    merged_df['struct'] = merged_df['struct'].fillna('Empty')

    # 셀의 최종 유형을 결정하는 함수
    def get_cell_type(row):
        # 건설 현장 우선순위 적용
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

    return merged_df


# --- 2단계: 지도 시각화 함수 ---

def draw_map(df, file_name, path=None, start_node=None, end_node=None, show_legend=True, show_plot=True):
    '''
    주어진 데이터프레임을 기반으로 지도를 시각화하여 이미지 파일로 저장합니다.

    Args:
        df (pandas.DataFrame): 지도에 표시할 데이터 (x, y, final_type 포함)
        file_name (str): 저장할 이미지 파일 이름 (예: 'map.png', 'map_final.png')
        path (list): (x, y) 튜플의 경로 리스트 (경로 시각화 시)
        start_node (tuple): 시작 노드의 (x, y) 좌표 (지도에 표시되지 않음)
        end_node (tuple): 끝 노드의 (x, y) 좌표 (지도에 표시되지 않음)
        show_legend (bool): 범례를 표시할지 여부
        show_plot (bool): 화면에 플롯을 표시할지 여부
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
    for index, row in df.iterrows():
        x, y = row['x'], row['y']
        cell_type = row['final_type']

        if cell_type == 'Apartment' or cell_type == 'Building':
            label = 'Apartment/Building'
            if label not in unique_labels:
                plt.plot(x, y, 'o', color='saddlebrown', markersize=20, label=label)
                unique_labels[label] = True
            else:
                plt.plot(x, y, 'o', color='saddlebrown', markersize=20)
        elif cell_type == 'BandalgomCoffee':
            label = 'Bandalgom Coffee'
            if label not in unique_labels:
                plt.plot(x, y, 's', color='green', markersize=20, label=label)
                unique_labels[label] = True
            else:
                plt.plot(x, y, 's', color='green', markersize=20)
        elif cell_type == 'MyHome':
            label = 'My Home'
            if label not in unique_labels:
                plt.plot(x, y, '^', color='green', markersize=20, label=label)
                unique_labels[label] = True
            else:
                plt.plot(x, y, '^', color='green', markersize=20)
        elif cell_type == 'ConstructionSite':
            label = 'Construction Site'
            # 건설 현장은 바로 옆 좌표와 살짝 겹쳐도 되므로, 마커 크기를 약간 크게 설정
            if label not in unique_labels:
                plt.plot(x, y, 's', color='gray', markersize=36, label=label, alpha=0.8)
                unique_labels[label] = True
            else:
                plt.plot(x, y, 's', color='gray', markersize=36, alpha=0.8)

    # 경로 플로팅
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        label = 'Shortest Path'
        if label not in unique_labels:
            plt.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=10, label=label)
            unique_labels[label] = True
        else:
            plt.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=10)

    plt.title('Area Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # X, Y 축 범위 설정 및 Y축 반전 ((1,1)이 좌측 상단이 되도록)
    plt.xlim(0.5, max_x + 0.5)
    plt.ylim(max_y + 0.5, 0.5)

    # 범례 표시
    if show_legend:
        legend_items = [
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.7, 
                         edgecolor='black', linewidth=0.5, label='Construction Site'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='saddlebrown', 
                      markersize=12, markeredgecolor='black', markeredgewidth=0.5, 
                      label='Apartment / Building'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkgreen', 
                      markersize=12, markeredgecolor='black', markeredgewidth=0.5,
                      label='Bandalgom Coffee'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='limegreen', 
                      markersize=14, markeredgecolor='black', markeredgewidth=0.5,
                      label='My Home'),
        ]
        
        if path:
            legend_items.append(
                plt.Line2D([0], [0], color='red', linewidth=3, alpha=0.8, label='Shortest Path')
            )
        
        ax.legend(handles=legend_items, loc='lower right', frameon=True, 
                 fancybox=True, shadow=True, fontsize=10)
    
    plt.xticks(list(range(1, max_x + 1)))
    plt.yticks(list(range(1, max_y + 1)))
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 파일 저장
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    print(f"{file_name} 파일이 저장되었습니다.")
    
    # 화면 표시 옵션
    if show_plot:
        try:
            plt.show()  # 윈도우에 플롯 표시
            print("플롯이 새 창에 표시되었습니다.")
        except Exception as e:
            print(f"화면 표시 실패: {e}")
            print("파일로만 저장됩니다.")
    else:
        plt.close()


# --- 3단계: 경로 탐색 및 메인 로직 ---

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
    frontier = [] # (f_cost, 노드) 튜플 저장
    frontier.append((0, start))
    frontier.sort() # 항상 정렬된 상태 유지

    came_from = {} # 경로 재구성을 위한 맵: {현재 노드: 이전 노드}
    g_cost = {start: 0} # 시작점에서 각 노드까지의 실제 비용
    f_cost = {start: _heuristic(start, goal)} # A* 비용 (g_cost + heuristic)

    while frontier:
        # 가장 작은 f_cost를 가진 노드를 추출
        current_f_cost, current = frontier.pop(0)

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

            new_g_cost = g_cost[current] + 1 # 이웃으로 이동하는 비용은 1

            if neighbor not in g_cost or new_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_g_cost
                f_cost[neighbor] = new_g_cost + _heuristic(neighbor, goal)
                
                # 새로운 노드를 frontier에 추가하고 정렬
                frontier.append((f_cost[neighbor], neighbor))
                frontier.sort()
                
                came_from[neighbor] = current
    return None # 경로를 찾을 수 없음

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

    all_permutations = []
    for i in range(len(elements)):
        m = elements[i]
        # 현재 요소를 제외한 나머지 요소들
        remaining_elements = elements[:i] + elements[i+1:]
        # 나머지 요소들의 순열을 재귀적으로 생성
        for p in _generate_permutations(remaining_elements):
            all_permutations.append([m] + p) # 현재 요소를 각 순열의 시작에 추가
    return all_permutations


def find_optimal_path_visiting_all_structures(grid_width, grid_height, start, end, structures_to_visit, impassable_cells):
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
    # 방문할 중간 구조물이 없는 경우, 단순히 시작점에서 끝점까지 A* 탐색
    if not structures_to_visit:
        return _a_star_search(grid_width, grid_height, start, end, impassable_cells)

    # 시작점과 끝점을 중간 방문 구조물에서 제외하고 순열을 위한 고유한 좌표만 정렬하여 사용
    intermediate_structures = sorted(list(set(s for s in structures_to_visit if s != start and s != end)))

    best_full_path = None
    min_total_length = float('inf')

    # 중간 구조물 방문 순서의 모든 순열 고려 (TSP 해결)
    # itertools.permutations 대신 사용자 정의 _generate_permutations 함수 사용
    for perm in _generate_permutations(intermediate_structures):
        current_path_sequence = [start] + list(perm) + [end] # 전체 순서
        current_total_length = 0
        current_full_path_nodes = []
        path_segments_possible = True

        for i in range(len(current_path_sequence) - 1):
            segment_start = current_path_sequence[i]
            segment_end = current_path_sequence[i+1]

            # A*를 사용하여 세그먼트 경로 찾기
            path_segment = _a_star_search(grid_width, grid_height, segment_start, segment_end, impassable_cells)

            if path_segment is None: # 경로를 찾을 수 없는 경우
                path_segments_possible = False
                break

            # 전체 경로에 세그먼트 추가 (다음 세그먼트의 시작 노드 중복 방지)
            if i == 0:
                current_full_path_nodes.extend(path_segment)
            else:
                current_full_path_nodes.extend(path_segment[1:]) # 첫 노드(이전 세그먼트의 끝 노드) 제외

            current_total_length += len(path_segment) - 1 # 각 단계는 길이에 1을 더함

        if path_segments_possible and current_total_length < min_total_length:
            min_total_length = current_total_length
            best_full_path = current_full_path_nodes

    return best_full_path


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    print("matplotlib 백엔드 테스트 중...")
    
    # 데이터 로드 및 전처리
    try:
        merged_df = load_and_process_data('area_category.csv', 'area_map.csv', 'area_struct.csv')
        print("데이터 로드 성공")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        exit()

    # 맵 크기(최대 x, y 좌표) 가져오기
    max_x = merged_df['x'].max()
    max_y = merged_df['y'].max()

    # 경로 탐색을 위한 통과 불가능한(건설 현장) 노드 집합 생성
    impassable_nodes = set()
    for index, row in merged_df[merged_df['final_type'] == 'ConstructionSite'].iterrows():
        impassable_nodes.add((row['x'], row['y']))


    # 내 집과 반달곰 커피 지점 좌표 찾기
    my_home_coords = merged_df[merged_df['final_type'] == 'MyHome'][['x', 'y']].values
    bandalgom_coffee_coords = merged_df[merged_df['final_type'] == 'BandalgomCoffee'][['x', 'y']].values

    if len(my_home_coords) == 0:
        print('Error: MyHome not found on the map.')
        exit()
    if len(bandalgom_coffee_coords) == 0:
        print('Error: Bandalgom Coffee not found on the map.')
        exit()

    start_node = tuple(my_home_coords[0])
    end_node = tuple(bandalgom_coffee_coords[0])

    # 방문해야 할 모든 접근 가능한 구조물 노드 찾기
    # 건설 현장을 제외한 아파트, 빌딩, 내 집, 반달곰 커피 지점
    accessible_structures = []
    for index, row in merged_df.iterrows():
        if row['final_type'] in ['Apartment', 'Building', 'MyHome', 'BandalgomCoffee']:
            coord = (row['x'], row['y'])
            # 해당 구조물 위치 자체가 건설 현장이 아닌 경우에만 추가
            if coord not in impassable_nodes:
                accessible_structures.append(coord)

    accessible_structures = list(set(accessible_structures)) # 중복 제거

    print(f'My Home (시작점): {start_node}')
    print(f'Bandalgom Coffee (도착점): {end_node}')
    print(f'방문해야 할 구조물 (건설 현장 제외): {accessible_structures}')

    # 최종 경로 계산
    print("최적 경로 계산 중...")
    final_path = find_optimal_path_visiting_all_structures(
        max_x, max_y, start_node, end_node, accessible_structures, impassable_nodes
    )

    if final_path:
        print(f'최단 경로 길이: {len(final_path) - 1} 단계')
        # 경로를 CSV로 저장
        path_df = pd.DataFrame(final_path, columns=['x', 'y'])
        path_df.to_csv('home_to_cafe_bonus.csv', index=False)
        print('home_to_cafe_bonus.csv 파일이 저장되었습니다.')

        # 경로가 표시된 최종 맵 그리기 및 표시
        print("최종 맵 생성 중...")
        draw_map(merged_df, 'map_final_bonus.png', path=final_path, 
                start_node=start_node, end_node=end_node, show_plot=True)
    else:
        print('지정된 모든 구조물을 방문하는 경로를 찾을 수 없습니다.')
        # 경로를 찾을 수 없는 경우, 경로가 없는 최종 맵 다시 그리기
        draw_map(merged_df, 'map_final_bonus.png', show_plot=True)
        print('map_final_bonus.png 파일이 저장되었습니다 (경로 없음).')
    
    print("프로그램이 완료되었습니다.")