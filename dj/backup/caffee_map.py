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
# 'category' 컬럼을 기준으로 병합하고, 'struct' 컬럼 이름을 'struct_name'으로 변경
area_struct_df = pd.merge(area_struct_df, area_category_df, on='category', how='left')
area_struct_df = area_struct_df.rename(columns={'struct': 'struct_name'})

# 3. 세 데이터를 하나의 DataFrame으로 병합하고, area 기준으로 정렬
# area_map_df와 수정된 area_struct_df를 병합합니다.
# area_map_df에는 'x', 'y'만 있고, area_struct_df에는 'x', 'y', 'category', 'area', 'struct_name'이 있습니다.
# 두 DataFrame을 'x'와 'y'를 기준으로 병합해야 합니다.
# 또한 area_map_df의 'ConstructionSite' 컬럼을 건설현장 여부를 나타내는 데 활용해야 합니다.
# 이전에 area_struct_df에 'area' 컬럼이 있으므로, 이 정보를 사용하여 병합합니다.

# 여기서는 area_map_df의 'x', 'y' 좌표에 매칭되는 area_struct_df의 정보를 가져오도록 합니다.
# 'map_id'는 area_map.csv에만 있는 것 같으므로, 실제 CSV 파일 구조를 다시 확인해야 합니다.
# 제공된 CSV 내용에는 'map_id' 컬럼이 보이지 않습니다.
# 만약 'map_id'가 없고 'x', 'y'로만 결합해야 한다면:
merged_df = pd.merge(area_map_df, area_struct_df, on=['x', 'y'], how='left')

# ConstructionSite 정보와 struct_name 정보가 겹칠 경우, ConstructionSite가 우선되도록 처리
# 'ConstructionSite' 컬럼이 1인 경우, '건설 현장'으로 struct_name을 덮어씁니다.
merged_df.loc[merged_df['ConstructionSite'] == 1, 'struct_name'] = '건설 현장'

# 'area' 기준으로 정렬 (오름차순)
merged_df = merged_df.sort_values(by='area')

print("\n--- 병합된 DataFrame (정렬 후 5개 행) ---")
print(merged_df.head())

# 4. area 1에 대한 데이터만 필터링해서 출력
area1_df = merged_df[merged_df['area'] == 1].copy() # SettingWithCopyWarning 방지를 위해 .copy() 사용

print("\n--- Area 1 필터링된 DataFrame ---")
print(area1_df)

# 5. (보너스) 구조물 종류별 요약 통계 리포트 출력
print("\n--- 구조물 종류별 요약 통계 ---")
structure_counts = area1_df['struct_name'].value_counts()
print(structure_counts)

# 결과를 CSV로 저장
area1_df.to_csv('area1_filtered_data.csv', index=False)
print("\nArea 1 필터링된 데이터가 'area1_filtered_data.csv'로 저장되었습니다.")