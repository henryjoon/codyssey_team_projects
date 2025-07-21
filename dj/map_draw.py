# map_draw.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # --- 1. 데이터 불러오기 ----------------------------------------------------
    map_df      = pd.read_csv('area_map.csv')
    struct_df   = pd.read_csv('area_struct.csv')
    category_df = pd.read_csv('area_category.csv')

    # 열 이름과 데이터 공백 제거
    category_df.columns = category_df.columns.str.strip()
    category_df['struct'] = category_df['struct'].str.strip()

    # category=0이 없으면 추가 (기본값)
    if not (category_df['category'] == 0).any():
        new_row = pd.DataFrame({'category': [0], 'struct': ['None']})
        category_df = pd.concat([new_row, category_df], ignore_index=True)

    # --- 2. 병합 (좌표 기준 → 구조물 번호 → 이름) ------------------------------
    merged = (
        map_df
        .merge(struct_df, on=['x', 'y'], how='left')
        .merge(category_df, on='category', how='left')
    )
    merged['struct'] = merged['struct'].fillna('None')

    # --- 3. 시각화 초기 설정 ---------------------------------------------------
    max_x, max_y = merged['x'].max(), merged['y'].max()
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_xlim(0.5, max_x + 0.5)
    ax.set_ylim(0.5, max_y + 0.5)
    ax.set_xticks(range(1, max_x + 1))
    ax.set_yticks(range(1, max_y + 1))
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.invert_yaxis()  # (1,1)이 좌측 상단
    ax.set_aspect('equal')

    # --- 4. 건설 현장 먼저 그리기 (바닥에 깔리게) ------------------------------
    construction = merged[merged['ConstructionSite'] == 1]
    for _, r in construction.iterrows():
        # 건설 현장은 살짝 겹쳐도 되도록 사각형 크기를 약간 크게
        ax.add_patch(plt.Rectangle(
            (r['x'] - 0.3, r['y'] - 0.3),
            0.6, 0.6,
            color='gray', alpha=0.7, zorder=1
        ))

    # --- 5. 구조물 나중에 그리기 (건설현장 위에 올라오게) -----------------------
    for _, r in merged.iterrows():
        if r['struct'] == 'None':        
            continue
        elif r['struct'] in ('Apartment', 'Building'):
            ax.plot(r['x'], r['y'], 'o', color='saddlebrown', alpha=0.9, 
                   markersize=20, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        elif r['struct'] == 'BandalgomCoffee':
            ax.plot(r['x'], r['y'], 's', color='darkgreen', alpha=0.9, 
                   markersize=20, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        elif r['struct'] == 'MyHome':
            ax.plot(r['x'], r['y'], '^', color='limegreen', alpha=0.9, 
                   markersize=22, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    # --- 6. 범례 추가 ----------------------------------------------------------
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
    ax.legend(handles=legend_items, loc='lower right', frameon=True, 
             fancybox=True, shadow=True, fontsize=10)
    
    # x축 눈금 라벨을 위쪽으로 이동
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    
    ax.set_title('Map', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    # x축 라벨도 위쪽으로 이동
    ax.xaxis.set_label_position('top')

    # --- 7. 저장 ---------------------------------------------------------------
    plt.tight_layout()
    plt.savefig('map.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()  # 실행 시 미리보기
    print(f'✅ 지도 저장 완료: {os.path.abspath("map.png")}')

if __name__ == '__main__':
    main()