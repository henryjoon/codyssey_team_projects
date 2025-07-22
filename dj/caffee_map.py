import pandas as pd


def load_data():
    ##load csv 
    map_df = pd.read_csv('area_map.csv')
    struct_df = pd.read_csv('area_struct.csv')
    category_df = pd.read_csv('area_category.csv')

    # struct 곰백 제거 ' struct'=>'struct'
    category_df.columns = category_df.columns.str.strip()

    return map_df, struct_df, category_df


def merge_data(map_df, struct_df, category_df):
    ## 병합 && 이름으로 변환
    # 구조물 이름 병합 => struct 추가
    struct_df = struct_df.merge(category_df, on = 'category', how = 'inner')
    del struct_df['category']  # category 열 제거, 필요 없으므로

    # map과 병합 => ConstructionSite 추가
    merged_df = map_df.merge(struct_df, on = ['x','y'], how = 'left')
    
    # area 기준 sort
    merged_df = merged_df.sort_values(by = 'area')
    return merged_df


def filter_area_one(df):
    ## area 1 필터링
    return df[df['area'] == 1].copy()


def summarize_by_structure(df):
    ## (보너스) 구조물 통계 요약
    print('\n[구조물 종류별 통계]')
    print(df['struct'].value_counts())


def main():
    map_df, struct_df, category_df = load_data()
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력

    
    # 출력
    print(f'[area_map.csv]\n{map_df.head()}')
    print(f'\n[area_struct.csv]\n{struct_df.head()}')
    print(f'\n[area_category]\n{category_df.head()}')
    
    # 병합
    merged_df = merge_data(map_df, struct_df, category_df)
    area1_df = filter_area_one(merged_df)
    merged_df.to_csv('merged.csv', index = False)
    #print(f'\n[merge]\n{merged_df}')

    #분석 => area별 반달곰커피 개수
    all_areas = merged_df['area'].unique()
    all_areas.sort()

    # 공백 제거한 struct로 비교
    coffee_counts = (
        merged_df[merged_df['struct'].str.strip() == 'BandalgomCoffee']
        .groupby('area')
        .size()
        .reindex(all_areas, fill_value = 0)
    )

    print('\n[area별 반달곰커피 개수]')
    for area, count in coffee_counts.items():
        print(f'area {area}: {count}개')


    # area 1 데이터 저장
    area1_df = area1_df.sort_values(by = ['x','y'])
    area1_df = area1_df[~((area1_df['ConstructionSite'] == 0) & (area1_df['struct'].isna()))]
    
    area1_df.to_csv('area1_filtered.csv', index = False)

    # area 1 데이터 출력
    print(f'\n[area 1 데이터]\n{area1_df}\n* 출력되지 않은 좌표에는 공사장도, 건물도 존재하지 않습니다.')



if __name__ == '__main__':
    main()