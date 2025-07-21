# map_draw.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def main():
    # --- 1. 데이터 불러오기 ----------------------------------------------------
    map_df      = pd.read_csv('area_map.csv')
    struct_df   = pd.read_csv('area_struct.csv')
    category_df = pd.read_csv('area_category.csv')

   # 열 이름 공백 제거
    category_df.columns = category_df.columns.str.strip()
    category_df['struct'] = category_df['struct'].str.strip()

    # category=0,None이 없으면 추가
    if not (category_df['category'] == 0).any():
        category_df = pd.concat([
            pd.DataFrame({'category': [0], 'struct': ['None']}),
            category_df
        ], ignore_index=True)

    # --- 2. 병합 (좌표 기준 → 구조물 번호 → 이름) ------------------------------
    merged = (
        map_df
        .merge(struct_df, on=['x', 'y'], how='left')
        .merge(category_df, on='category', how='left')
    )
    merged['struct'] = merged['struct'].fillna('None')

    # --- 3. 시각화 초기 설정 ---------------------------------------------------
    max_x, max_y = merged['x'].max(), merged['y'].max()
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(0.5, max_x + 0.5)
    ax.set_ylim(0.5, max_y + 0.5)
    ax.set_xticks(np.arange(1, max_x + 1))
    ax.set_yticks(np.arange(1, max_y + 1))
    ax.grid(True)
    ax.invert_yaxis()  # (1,1)이 좌측 상단
    # --- 4. 구조물 먼저 그리기 (건설현장 위에 올라오게) -------------------------
    for _, r in merged.iterrows():
        if r['struct'] == 'None':        
            continue
        elif r['struct'] in ('Apartment', 'Building'):
            ax.plot(r['x'], r['y'], 'o', color='brown', alpha=0.8, markersize=18)
        elif r['struct'] == 'BandalgomCoffee':
            ax.plot(r['x'], r['y'], 's', color='green', alpha=0.9, markersize=18)
        elif r['struct'] == 'MyHome':
            ax.plot(r['x'], r['y'], '^', color='green', alpha=0.9, markersize=18)
    # --- 5. 건설 현장 사각형을 반투명 회색으로 덮기 ----------------------------
    construction = merged[merged['ConstructionSite'] == 1]
    for _, r in construction.iterrows():
        ax.add_patch(plt.Rectangle(
            (r['x'] - 0.25, r['y'] - 0.25),
            0.5, 0.5,
            color='gray', alpha=0.9  # 반투명 처리
            ))

    # --- 6. 범례 추가 ----------------------------------------------------------
    legend_items = [
        mpatches.Patch(color='gray', alpha=0.4, label='Construction Site'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='saddlebrown', markersize=10, label='Apartment / Building'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Bandalgom Coffee'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='My Home'),
    ]
    ax.legend(handles=legend_items, loc='lower right')
    ax.set_title('Map')

    # --- 7. 저장 ---------------------------------------------------------------
    plt.savefig('map.png', bbox_inches='tight')
    print(f'✅ 지도 저장 완료: {os.path.abspath('map.png')}')

if __name__ == '__main__':
    main()
