import pandas as pd
def main():
# --- 1. 데이터 불러오기 --------------------------------------------------
    df = pd.read_csv('merged.csv')
    df['struct'] = df['struct'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # --- 5. 구조물별 통계 요약 ------------------------------------------------
    summary = (
        df
        .groupby('struct')
        .agg(
            total_points       = ('struct', 'size'),
            area1              = ('area', lambda a: (a == 1).sum()),
            area2              = ('area', lambda a: (a == 2).sum()),
            area3              = ('area', lambda a: (a == 3).sum()),
            area4              = ('area', lambda a: (a == 4).sum()),
            construction_sites  = ('ConstructionSite', 'sum'),
            construction_ratio  = ('ConstructionSite', 'mean'),
            center_x           = ('x', 'mean'),
            center_y           = ('y', 'mean'),
            min_x              = ('x', 'min'),
            max_x              = ('x', 'max'),
            min_y              = ('y', 'min'),
            max_y              = ('y', 'max'),

        )
        .sort_values('total_points', ascending = False)
    )

    # --- 6. 소수점 정리 -------------------------------------------------------
    summary['construction_ratio'] = summary['construction_ratio'].round(2)
    summary['center_x'] = summary['center_x'].round(1)
    summary['center_y'] = summary['center_y'].round(1)

    # --- 7. 출력 및 저장 -------------------------------------------------------
    print('=== 구조물별 통계 요약 ===')
    print(summary)

    summary.to_csv('struct_summary_report.csv')
    print('\n✅ 저장 완료: struct_summary_report.csv')


if __name__ == '__main__':
    main()