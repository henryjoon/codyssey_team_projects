import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations
import math

# Read the CSV files
try:
    area_struct_df = pd.read_csv('area_struct.csv')
    area_map_df = pd.read_csv('area_map.csv')
    area_category_df = pd.read_csv('area_category.csv')
except FileNotFoundError as e:
    print(f"Error reading CSV files: {e}")
    print("Please make sure the CSV files are in the current directory.")
    exit(1)

# Clean the category data
area_category_df.columns = area_category_df.columns.str.strip()
area_category_df['category'] = area_category_df['category'].astype(int)
area_category_df['struct'] = area_category_df['struct'].str.strip()

print("Category mapping:")
print(area_category_df)

# Find all structures (non-zero categories) excluding construction sites
structures = area_struct_df[area_struct_df['category'] != 0].copy()
print(f"\nFound {len(structures)} structures:")
print(structures)

# Get structure positions
structure_positions = [(row['x'], row['y']) for _, row in structures.iterrows()]
print(f"\nStructure positions: {structure_positions}")

# Calculate distance between two points
def calculate_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Nearest neighbor algorithm for TSP approximation
def nearest_neighbor_tsp(positions):
    if len(positions) <= 1:
        return positions
    
    unvisited = positions[1:]  # Start from first position
    path = [positions[0]]
    current = positions[0]
    
    while unvisited:
        # Find nearest unvisited position
        nearest_pos = min(unvisited, key=lambda pos: calculate_distance(current, pos))
        path.append(nearest_pos)
        current = nearest_pos
        unvisited.remove(nearest_pos)
    
    return path

# Find optimal path
optimal_path = nearest_neighbor_tsp(structure_positions)
print(f"\nOptimal path: {optimal_path}")

# Calculate total distance
total_distance = sum(calculate_distance(optimal_path[i], optimal_path[i+1]) 
                    for i in range(len(optimal_path)-1))
print(f"Total path distance: {total_distance:.2f}")

# Create the visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Set up the grid - invert y-axis so (1,1) is top-left
ax.set_xlim(0.5, 15.5)
ax.set_ylim(15.5, 0.5)  # Inverted y-axis
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Map with Optimized Path', fontsize=16, fontweight='bold')
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)

ax.xaxis.set_label_position('top')
# Set ticks
ax.set_xticks(range(1, 16))
ax.set_yticks(range(1, 16))
# x축 눈금을 위쪽으로
ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
# Draw construction sites (gray squares)
construction_sites = area_map_df[area_map_df['ConstructionSite'] == 1]
for _, site in construction_sites.iterrows():
    rect = patches.Rectangle((site['x']-0.3, site['y']-0.3), 0.6, 0.6, 
                           linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.7)
    ax.add_patch(rect)

# Color mapping for structures
color_map = {1: 'brown', 2: 'brown', 3: 'green', 4: 'green'}
shape_map = {1: 'o', 2: 'o', 3: '^', 4: 's'}
size_map = {1: 400, 2: 400, 3: 400, 4: 300}

# Draw structures
for _, structure in structures.iterrows():
    category = structure['category']
    color = color_map.get(category, 'black')
    shape = shape_map.get(category, 'o')
    size = size_map.get(category, 200)
    
    ax.scatter(structure['x'], structure['y'], c=color, s=size, marker=shape, 
              edgecolor='black', linewidth=1.5, alpha=0.8,  zorder=5)

# Draw the optimized path with red lines
if len(optimal_path) > 1:
    path_x = [pos[0] for pos in optimal_path]
    path_y = [pos[1] for pos in optimal_path]
    
    ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8, zorder=4, label='Optimized Path')
    
    # Add arrows to show direction
    for i in range(len(optimal_path)-1):
        dx = optimal_path[i+1][0] - optimal_path[i][0]
        dy = optimal_path[i+1][1] - optimal_path[i][1]
        ax.annotate('', xy=optimal_path[i+1], xytext=optimal_path[i],
                   arrowprops=dict(arrowstyle='->', color='red', lw=2), zorder=6)

# Add legend
legend_elements = [
    patches.Patch(color='lightgray', label='Construction Site'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', 
               markersize=10, label='Apartment / Building'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=8, label='Bandalgom Coffee'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
               markersize=10, label='My Home'),
    plt.Line2D([0], [0], color='red', linewidth=3, label='Optimized Path')
]

ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))

# Add path information as text
info_text = f"Total structures visited: {len(optimal_path)}\nTotal path distance: {total_distance:.2f}"
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save the figure
plt.savefig('map_all_final.png', dpi=300, bbox_inches='tight')
print(f"\nMap saved as 'map_all_final.png'")

plt.show()

# Print summary
print("\n" + "="*50)
print("PATH OPTIMIZATION SUMMARY")
print("="*50)
print(f"Structures to visit: {len(structure_positions)}")
print(f"Optimal path order:")
for i, pos in enumerate(optimal_path):
    structure_info = structures[(structures['x'] == pos[0]) & (structures['y'] == pos[1])].iloc[0]
    category_name = area_category_df[area_category_df['category'] == structure_info['category']]['struct'].iloc[0]
    print(f"  {i+1}. ({pos[0]}, {pos[1]}) - {category_name}")
print(f"Total distance: {total_distance:.2f} units")
print("="*50)
