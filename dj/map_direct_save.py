# map_direct_save.py
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import os

def load_and_merge_data():
    """데이터 불러오기 및 병합"""
    map_df = pd.read_csv('area_map.csv')
    struct_df = pd.read_csv('area_struct.csv')
    category_df = pd.read_csv('area_category.csv')

    # 열 이름과 데이터 공백 제거
    category_df.columns = category_df.columns.str.strip()
    category_df['struct'] = category_df['struct'].str.strip()

    # category=0이 없으면 추가
    if not (category_df['category'] == 0).any():
        new_row = pd.DataFrame({'category': [0], 'struct': ['None']})
        category_df = pd.concat([new_row, category_df], ignore_index=True)

    # 병합
    merged = (
        map_df
        .merge(struct_df, on=['x', 'y'], how='left')
        .merge(category_df, on='category', how='left')
    )
    merged['struct'] = merged['struct'].fillna('None')
    
    return merged

def find_positions(merged_df):
    """시작점(내 집)과 도착점(반달곰 커피) 위치 찾기"""
    home_pos = merged_df[merged_df['struct'] == 'MyHome'][['x', 'y']].values
    cafe_pos = merged_df[merged_df['struct'] == 'BandalgomCoffee'][['x', 'y']].values
    
    if len(home_pos) == 0:
        raise ValueError("내 집 위치를 찾을 수 없습니다!")
    if len(cafe_pos) == 0:
        raise ValueError("반달곰 커피 위치를 찾을 수 없습니다!")
    
    # 첫 번째 위치 사용 (여러 개가 있을 경우)
    start = tuple(home_pos[0])
    goals = [tuple(pos) for pos in cafe_pos]
    
    return start, goals

def bfs_shortest_path(merged_df, start, goals):
    """BFS를 사용한 최단 경로 탐색"""
    max_x, max_y = merged_df['x'].max(), merged_df['y'].max()
    
    # 건설현장 위치들을 집합으로 저장 (빠른 검색을 위해)
    construction_sites = set()
    for _, row in merged_df[merged_df['ConstructionSite'] == 1].iterrows():
        construction_sites.add((row['x'], row['y']))
    
    # BFS 초기화
    queue = deque([(start, [start])])  # (현재위치, 경로)
    visited = {start}
    
    # 4방향 이동 (상, 하, 좌, 우)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (x, y), path = queue.popleft()
        
        # 목표 지점 중 하나에 도달했는지 확인
        if (x, y) in goals:
            return path
        
        # 4방향으로 이동
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # 경계 확인
            if nx < 1 or nx > max_x or ny < 1 or ny > max_y:
                continue
            
            # 이미 방문했거나 건설현장인 경우 스킵
            if (nx, ny) in visited or (nx, ny) in construction_sites:
                continue
            
            visited.add((nx, ny))
            queue.append(((nx, ny), path + [(nx, ny)]))
    
    return None  # 경로를 찾을 수 없는 경우

def save_path_to_csv(path, filename='home_to_cafe.csv'):
    """경로를 CSV 파일로 저장"""
    path_df = pd.DataFrame(path, columns=['x', 'y'])
    path_df.to_csv(filename, index=False)
    print(f'✅ 경로 저장 완료: {os.path.abspath(filename)}')
    return path_df

def visualize_map_with_path(merged_df, path):
    """지도와 경로 시각화"""
    max_x, max_y = merged_df['x'].max(), merged_df['y'].max()
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_xlim(0.5, max_x + 0.5) # x눈금 한계치 설정
    ax.set_ylim(0.5, max_y + 0.5) # y눈금 한계치 설정
    ax.set_xticks(range(1, max_x + 1)) #x 눈금 표시
    ax.set_yticks(range(1, max_y + 1))
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # x축 눈금을 위쪽으로
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    # 건설 현장 먼저 그리기
    construction = merged_df[merged_df['ConstructionSite'] == 1]
    for _, r in construction.iterrows():
        ax.add_patch(plt.Rectangle(
            (r['x'] - 0.35, r['y'] - 0.35),
            0.7, 0.7,
            color='gray', alpha=0.7, zorder=1
        ))

    # 구조물 그리기
    for _, r in merged_df.iterrows():  # 각행을 index, 시리즈로 반환
        if r['struct'] == 'None':        
            continue
        elif r['struct'] in ('Apartment', 'Building'):
            ax.plot(r['x'], r['y'], 'o', color='saddlebrown', alpha=0.9, 
                   markersize=16, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        elif r['struct'] == 'BandalgomCoffee':
            ax.plot(r['x'], r['y'], 's', color='darkgreen', alpha=0.9, 
                   markersize=16, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        elif r['struct'] == 'MyHome':
            ax.plot(r['x'], r['y'], '^', color='limegreen', alpha=0.9, 
                   markersize=18, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    # 최단 경로 그리기 (빨간 선)
    if path and len(path) > 1:
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8, zorder=2, label='Shortest Path')
        
        # 경로 점들 표시
        ax.plot(path_x, path_y, 'ro', markersize=4, alpha=0.6, zorder=2)

    # 범례 추가
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
    
    ax.set_title('Map with Shortest Path', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.xaxis.set_label_position('top')

    # 저장
    plt.tight_layout()
    plt.savefig('map_final.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f'✅ 최종 지도 저장 완료: {os.path.abspath("map_final.png")}')

def main():
    """메인 함수"""
    print("🗺️  지도 데이터 로딩 중...")
    merged_df = load_and_merge_data()
    
    print("📍 시작점과 도착점 찾는 중...")
    start, goals = find_positions(merged_df)
    print(f"   시작점 (내 집): {start}")
    print(f"   도착점 (반달곰 커피): {goals}")
    
    print("🔍 최단 경로 탐색 중 (BFS 알고리즘)...")
    path = bfs_shortest_path(merged_df, start, goals)
    
    if path:
        print(f"✅ 최단 경로 발견! 총 {len(path)}개 지점")
        print(f"   경로 길이: {len(path) - 1}칸")
        
        # 경로를 CSV로 저장
        save_path_to_csv(path)
        
        # 시각화
        print("🎨 지도 시각화 중...")
        visualize_map_with_path(merged_df, path)
        
    else:
        print("❌ 경로를 찾을 수 없습니다! 건설현장으로 인해 막혔을 수 있습니다.")

if __name__ == '__main__':
    main()