#!/usr/bin/env python3
"""
四元数变化曲线绘制脚本
读取 df_quaternion.csv 并生成曲线图
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_csv(filepath):
    """读取CSV文件"""
    time = []
    qw, qx, qy, qz = [], [], [], []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time.append(float(row['time']))
            qw.append(float(row['qw']))
            qx.append(float(row['qx']))
            qy.append(float(row['qy']))
            qz.append(float(row['qz']))
    
    return np.array(time), np.array(qw), np.array(qx), np.array(qy), np.array(qz)

def plot_quaternions(csv_path, output_path):
    """绘制四元数曲线"""
    # 读取数据
    t, qw, qx, qy, qz = read_csv(csv_path)
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Quaternion Components vs Time (Lemniscate Trajectory)', fontsize=14, fontweight='bold')
    
    # 绘制各分量
    data = [(qw, '$q_w$', '#1f77b4'), (qx, '$q_x$', '#ff7f0e'), 
            (qy, '$q_y$', '#2ca02c'), (qz, '$q_z$', '#d62728')]
    
    for idx, (q, label, color) in enumerate(data):
        ax = axes[idx // 2, idx % 2]
        ax.plot(t, q, color=color, linewidth=1.2, label=label)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label} Component', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([-1.1, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # 绘制合并图
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, qw, label='$q_w$', linewidth=1.5, color='#1f77b4')
    ax.plot(t, qx, label='$q_x$', linewidth=1.5, color='#ff7f0e')
    ax.plot(t, qy, label='$q_y$', linewidth=1.5, color='#2ca02c')
    ax.plot(t, qz, label='$q_z$', linewidth=1.5, color='#d62728')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Quaternion Value', fontsize=12)
    ax.set_title('All Quaternion Components vs Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([-1.1, 1.1])
    
    combined_path = output_path.replace('.png', '_combined.png')
    plt.tight_layout()
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close()

if __name__ == '__main__':
    # 文件路径
    csv_path = r'd:\fly\homework\solutions\df_quaternion.csv'
    output_path = r'd:\fly\homework\solutions\quaternion_plot.png'
    
    if os.path.exists(csv_path):
        plot_quaternions(csv_path, output_path)
        print("Done!")
    else:
        print(f"Error: {csv_path} not found!")
