import pandas as pd

# 1. 파일 불러오기 및 내용 출력
try:
    area_map_df = pd.read_csv('area_map.csv')
    area_struct_df = pd.read_csv('area_struct.csv')
    area_category_df = pd.read_csv('area_category.csv')

    print("--- area_map.csv 내용 ---")
    print(area_map_df.head())
    print("\n--- area_struct.csv 내용 ---")
    print(area_struct_df.head())
    print("\n--- area_category.csv 내용 ---")
    print(area_category_df.head())

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure all CSV files are in the same directory.")
    exit()

# 2. 구조물 ID를 area_category.csv 기준으로 이름으로 변환
# area_struct_df와 area_category_df를 'struct_id' 기준으로 병합
# merge 함수 사용 시, 동일한 컬럼이 있으면 _x, _y가 붙기 때문에 미리 이름을 변경한다.
area_struct_df = pd.merge(area_struct_df, area_category_df, on='struct_id', how='left')
# 변환 후 'struct_id' 컬럼은 더 이상 필요 없으므로 제거하거나, 'struct_name'으로 이름을 변경할 수 있다.
# 여기서는 'struct_name'을 사용하고, 기존 'struct_id'는 유지한다.

# 3. 세 데이터를 하나의 DataFrame으로 병합하고, area 기준으로 정렬
# area_map_df와 area_struct_df를 'map_id' 기준으로 병합
merged_df = pd.merge(area_map_df, area_struct_df, on='map_id', how='left')

# 'area' 기준으로 정렬 (오름차순)
merged_df = merged_df.sort_values(by='area')

print("\n--- 병합된 DataFrame (정렬 전 5개 행) ---")
print(merged_df.head())

# 4. area 1에 대한 데이터만 필터링해서 출력
area1_df = merged_df[merged_df['area'] == 1].copy() # SettingWithCopyWarning 방지를 위해 .copy() 사용

print("\n--- Area 1 필터링된 DataFrame ---")
print(area1_df)


# 5. (보너스) 구조물 종류별 요약 통계 리포트 출력
print("\n--- 구조물 종류별 요약 통계 ---")
structure_counts = area1_df['struct_name'].value_counts()
print(structure_counts)

# 결과를 CSV로 저장 (요구사항에는 없으나, 나중에 확인 용이성을 위해 저장)
area1_df.to_csv('area1_filtered_data.csv', index=False)
print("\nArea 1 필터링된 데이터가 'area1_filtered_data.csv'로 저장되었습니다.")

# 최종적으로 'caffee_map.py' 스크립트로 저장될 내용.
# 실제 파일 저장 로직은 여기에 포함시키지 않고, 스크립트 실행 결과로 간주
# 이 코드를 caffee_map.py 파일로 저장합니다.