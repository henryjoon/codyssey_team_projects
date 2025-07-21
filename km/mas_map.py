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
    struct_df = struct_df.merge(category_df, on='category', how='left')

    # map과 병합 => ConstructionSite 추가
    merged_df = map_df.merge(struct_df, on=['x','y'], how='left')
    
    # area 기준 sort
    merged_df = merged_df.sort_values(by='area')
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
    
    # 출력(5행만)
    print(f'[area_map.csv]\n{map_df.head()}')
    print(f'\n[area_struct.csv]\n{struct_df.head()}')
    print(f'\n[area_category]\n{category_df.head()}')
    
    # 병합
    merged_df = merge_data(map_df, struct_df, category_df)
    area1_df = filter_area_one(merged_df)
    merged_df.to_csv('merged.csv', index=False)
    print(f'\n[merge]\n{merged_df}')

    #분석 => area별 반달곰커피 개수
    coffee_counts = merged_df[merged_df['struct'] == ' BandalgomCoffee'].groupby('area').size()

    print('\n[area별 반달곰커피 개수]')
    for area, count in coffee_counts.items():
        print(f"area {area}: {count}개\n")


    # area 1 데이터 저장
    area1_df = area1_df.sort_values(by=['x','y'])
    area1_df.to_csv('area1_filtered.csv', index=False)

    # area 1 데이터 출력
    print(f'\n[area 1 데이터]\n{area1_df}')

    # (보너스) 구조물 종류별 통계 출력 (전체 지역)
    summarize_by_structure(merged_df)


if __name__ == '__main__':
    main()
