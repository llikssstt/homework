#!/usr/bin/env python3
"""绘制Task1末端执行器四元数曲线"""
import matplotlib.pyplot as plt
import numpy as np
import csv

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_csv(filepath):
    t, qx, qy, qz, qw = [], [], [], [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row['t']))
            qx.append(float(row['qx']))
            qy.append(float(row['qy']))
            qz.append(float(row['qz']))
            qw.append(float(row['qw']))
    return np.array(t), np.array(qx), np.array(qy), np.array(qz), np.array(qw)

csv_path = r'd:\fly\homework\solutions\task1_quaternion.csv'
output_path = r'd:\fly\homework\solutions\task1_quaternion_plot.png'

t, qx, qy, qz, qw = read_csv(csv_path)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t, qw, label='$q_w$', linewidth=1.5, color='#1f77b4')
ax.plot(t, qx, label='$q_x$', linewidth=1.5, color='#ff7f0e')
ax.plot(t, qy, label='$q_y$', linewidth=1.5, color='#2ca02c')
ax.plot(t, qz, label='$q_z$', linewidth=1.5, color='#d62728')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Quaternion Value', fontsize=12)
ax.set_title('End-Effector Quaternion in World Frame (Cone Motion)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim([-1.1, 1.1])

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")
