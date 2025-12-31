/**
 * @file df_quaternion_node.cpp
 * @brief 四旋翼微分平坦性 - 姿态四元数计算 (独立版本，不需要ROS)
 * 
 * 根据给定的双纽线轨迹，利用微分平坦性理论推导无人机姿态四元数
 * 
 * 轨迹方程（世界坐标系）：
 *   x = 10*cos(t) / (1 + sin²(t))
 *   y = 10*sin(t)*cos(t) / (1 + sin²(t))
 *   z = 10
 *   t ∈ [0, 2π]
 * 
 * 偏航角 ψ 始终与速度方向对齐
 */

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>

using namespace Eigen;
using namespace std;

// 物理常数
const double g = 9.81;  // 重力加速度

/**
 * @brief 计算给定时刻的位置
 */
Vector3d getPosition(double t) {
    double sin_t = sin(t);
    double cos_t = cos(t);
    double denom = 1.0 + sin_t * sin_t;
    
    double x = 10.0 * cos_t / denom;
    double y = 10.0 * sin_t * cos_t / denom;
    double z = 10.0;
    
    return Vector3d(x, y, z);
}

/**
 * @brief 计算给定时刻的速度（位置的一阶导数）
 */
Vector3d getVelocity(double t) {
    double sin_t = sin(t);
    double cos_t = cos(t);
    double sin2_t = sin_t * sin_t;
    double cos2_t = cos_t * cos_t;
    double denom = 1.0 + sin2_t;
    double denom2 = denom * denom;
    
    double dx = (-10.0 * sin_t * denom - 10.0 * cos_t * 2.0 * sin_t * cos_t) / denom2;
    double dy = (10.0 * (cos2_t - sin2_t) * denom - 10.0 * sin_t * cos_t * 2.0 * sin_t * cos_t) / denom2;
    double dz = 0.0;
    
    return Vector3d(dx, dy, dz);
}

/**
 * @brief 计算给定时刻的加速度（数值微分）
 */
Vector3d getAcceleration(double t) {
    double dt = 1e-6;
    Vector3d v1 = getVelocity(t - dt);
    Vector3d v2 = getVelocity(t + dt);
    return (v2 - v1) / (2.0 * dt);
}

/**
 * @brief 根据微分平坦性计算姿态四元数
 */
Quaterniond computeQuaternion(double t) {
    Vector3d vel = getVelocity(t);
    Vector3d acc = getAcceleration(t);
    Vector3d e3(0, 0, 1);
    
    // 1. 机体z轴方向（推力方向）: z_B = (a + g*e3) / ||a + g*e3||
    Vector3d thrust_direction = acc + g * e3;
    if (thrust_direction.norm() < 1e-6) {
        thrust_direction = e3;
    }
    Vector3d z_B = thrust_direction.normalized();
    
    // 2. 偏航角（与速度方向对齐）
    double psi = atan2(vel(1), vel(0));
    
    // 3. 期望x轴方向（水平面内）
    Vector3d x_C(cos(psi), sin(psi), 0);
    
    // 4. 机体y轴: y_B = z_B × x_C
    Vector3d y_B_temp = z_B.cross(x_C);
    if (y_B_temp.norm() < 1e-6) {
        y_B_temp = z_B.cross(Vector3d(0, 1, 0));
        if (y_B_temp.norm() < 1e-6) {
            y_B_temp = z_B.cross(Vector3d(1, 0, 0));
        }
    }
    Vector3d y_B = y_B_temp.normalized();
    
    // 5. 机体x轴: x_B = y_B × z_B
    Vector3d x_B = y_B.cross(z_B);
    x_B.normalize();
    
    // 6. 旋转矩阵 R = [x_B, y_B, z_B]
    Matrix3d R;
    R.col(0) = x_B;
    R.col(1) = y_B;
    R.col(2) = z_B;
    
    // 7. 转换为四元数并归一化
    Quaterniond q(R);
    q.normalize();
    
    // 8. 确保 qw >= 0
    if (q.w() < 0) {
        q.coeffs() = -q.coeffs();
    }
    
    return q;
}

int main(int argc, char** argv) {
    cout << "========================================" << endl;
    cout << "Quadrotor Differential Flatness" << endl;
    cout << "Quaternion Calculation" << endl;
    cout << "========================================" << endl;
    
    // 输出文件
    string output_path = "df_quaternion.csv";
    ofstream outFile(output_path);
    
    if (!outFile.is_open()) {
        cerr << "Error: Failed to open output file!" << endl;
        return -1;
    }
    
    // CSV头
    outFile << "time,qw,qx,qy,qz" << endl;
    
    // 时间参数
    double t_start = 0.0;
    double t_end = 2.0 * M_PI;
    double dt = 0.02;
    
    int count = 0;
    
    for (double t = t_start; t <= t_end + 1e-9; t += dt) {
        Quaterniond q = computeQuaternion(t);
        
        outFile << fixed << setprecision(2) << t << ","
                << fixed << setprecision(7) << q.w() << ","
                << fixed << setprecision(7) << q.x() << ","
                << fixed << setprecision(7) << q.y() << ","
                << fixed << setprecision(7) << q.z() << endl;
        
        count++;
    }
    
    outFile.close();
    
    cout << "Calculation complete! Total " << count << " samples." << endl;
    cout << "Results saved to: " << output_path << endl;
    
    // 打印示例
    cout << "\n===== Sample Results =====" << endl;
    double samples[] = {0.0, M_PI/2, M_PI, 3*M_PI/2, 2*M_PI};
    for (double t : samples) {
        Quaterniond q = computeQuaternion(t);
        cout << "t=" << fixed << setprecision(2) << t 
             << ": qw=" << setprecision(4) << q.w() 
             << ", qx=" << q.x() 
             << ", qy=" << q.y() 
             << ", qz=" << q.z() << endl;
    }
    
    return 0;
}
