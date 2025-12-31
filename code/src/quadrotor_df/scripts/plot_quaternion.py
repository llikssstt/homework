#!/usr/bin/env python3
"""
四元数变化曲线绘制脚本

读取 df_quaternion.csv 文件并绘制四元数分量随时间的变化曲线
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_quaternion_curves(csv_path, output_path=None):
    """
    绘制四元数分量随时间的变化曲线
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出图片路径（可选）
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Quadrotor Attitude Quaternion vs Time (Lemniscate Trajectory)', fontsize=14)
    
    # 时间数据
    t = df['time']
    
    # 绘制每个四元数分量
    components = ['qw', 'qx', 'qy', 'qz']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['$q_w$', '$q_x$', '$q_y$', '$q_z$']
    
    for idx, (comp, color, label) in enumerate(zip(components, colors, labels)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(t, df[comp], color=color, linewidth=1.5, label=label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim([0, 2*np.pi])
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.show()
    
    # 绘制所有分量在同一图中
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for comp, color, label in zip(components, colors, labels):
        ax2.plot(t, df[comp], color=color, linewidth=1.5, label=label)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Quaternion Component')
    ax2.set_title('All Quaternion Components vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, 2*np.pi])
    
    plt.tight_layout()
    
    if output_path:
        combined_path = output_path.replace('.png', '_combined.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"Combined figure saved to: {combined_path}")
    
    plt.show()

def main():
    # 尝试多个可能的路径
    possible_paths = [
        '/home/stuwork/MRPC-2025-homework/code/src/quadrotor_df/df_quaternion.csv',
        './df_quaternion.csv',
        '../df_quaternion.csv',
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        print("Error: df_quaternion.csv not found!")
        print("Please run the df_quaternion_node first to generate the data.")
        return
    
    print(f"Reading data from: {csv_path}")
    
    # 输出图片路径
    output_path = csv_path.replace('.csv', '_plot.png')
    
    # 绘制曲线
    plot_quaternion_curves(csv_path, output_path)

if __name__ == '__main__':
    main()
