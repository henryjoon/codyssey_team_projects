import pandas as pd

# 1. CSV 파일 불러오기
area_map = pd.read_csv("dataFile/area_map.csv")
area_struct = pd.read_csv("dataFile/area_struct.csv")
area_category = pd.read_csv("dataFile/area_category.csv")

# 2. 공백 제거
area_map.columns = area_map.columns.str.strip()
area_struct.columns = area_struct.columns.str.strip()
area_category.columns = area_category.columns.str.strip()
area_category["struct"] = area_category["struct"].str.strip()

# 3. category 기준으로 struct 이름 붙이기
struct_with_name = area_struct.merge(area_category, on="category", how="left")

# 4. 위치 기반 병합
merged = area_map.merge(struct_with_name, on=["x", "y"], how="left")

# ✅ 5. 병합 결과를 CSV로 저장 (모든 셀 포함!)
merged.to_csv("merged_all.csv", index=False)

# ✅ 6. 터미널 출력용 데이터는 조건 필터링 (공사장도 건물도 없으면 제외)
filtered = merged[~(merged["struct"].isna() & (merged["ConstructionSite"] == 0))]

# ✅ 7. 구조물 기준 개수 출력
category_counts = filtered["struct"].dropna().value_counts()
print("\n[전체 지역 구조물 종류별 개수]")
print(category_counts)
