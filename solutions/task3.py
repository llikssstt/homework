import csv
import math
from pathlib import Path
from typing import List, Tuple

SOLUTIONS_DIR = Path(__file__).resolve().parent
ROOT = SOLUTIONS_DIR.parent

# ----------------------------- math helpers ----------------------------- #


def quat_to_rot(q: Tuple[float, float, float, float]) -> List[List[float]]:
    # 将四元数规范化后生成旋转矩阵（世界 -> 机体）。
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        norm = 1.0
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def rot_transpose(R: List[List[float]]) -> List[List[float]]:
    # 转置旋转矩阵（求逆）
    return [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]


def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    # 矩阵乘法
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]


def rot_to_quat(R: List[List[float]]) -> Tuple[float, float, float, float]:
    # 旋转矩阵转四元数（x, y, z, w）
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        S = math.sqrt(t + 1.0) * 2
        w = 0.25 * S
        x = (R[2][1] - R[1][2]) / S
        y = (R[0][2] - R[2][0]) / S
        z = (R[1][0] - R[0][1]) / S
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        S = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        w = (R[2][1] - R[1][2]) / S
        x = 0.25 * S
        y = (R[0][1] + R[1][0]) / S
        z = (R[0][2] + R[2][0]) / S
    elif R[1][1] > R[2][2]:
        S = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        w = (R[0][2] - R[2][0]) / S
        x = (R[0][1] + R[1][0]) / S
        y = 0.25 * S
        z = (R[1][2] + R[2][1]) / S
    else:
        S = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        w = (R[1][0] - R[0][1]) / S
        x = (R[0][2] + R[2][0]) / S
        y = (R[1][2] + R[2][1]) / S
        z = 0.25 * S
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return x, y, z, w


def normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # 向量归一化
    x, y, z = vec
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return 0.0, 0.0, 0.0
    return x / n, y / n, z / n


def cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # 计算向量叉积
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


# ----------------------------- task 1 ----------------------------- #


def compute_task1():
    tracking_path = ROOT / "documents" / "tracking.csv"
    alpha = math.pi / 12
    omega = 0.5
    rows = []
    with tracking_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = float(r["t"])
            q_body_world = (
                float(r["qx"]),
                float(r["qy"]),
                float(r["qz"]),
                float(r["qw"]),
            )
            R_bw = quat_to_rot(q_body_world)  # world -> body
            R_wb = rot_transpose(R_bw)  # body -> world

            wt = omega * t
            ca = math.cos(alpha)
            sa = math.sin(alpha)
            cw = math.cos(wt)
            sw = math.sin(wt)
            R_bd = [
                [cw, -sw * ca, sw * sa],
                [sw, cw * ca, -cw * sa],
                [0.0, sa, ca],
            ]  # D relative to B

            R_wd = mat_mul(R_wb, R_bd)
            q = rot_to_quat(R_wd)
            rows.append((t, *q))

    out_path = SOLUTIONS_DIR / "task1_quaternion.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "qx", "qy", "qz", "qw"])
        for t, qx, qy, qz, qw in rows:
            writer.writerow([f"{t:.2f}", f"{qx:.8f}", f"{qy:.8f}", f"{qz:.8f}", f"{qw:.8f}"])
    create_svg(rows, SOLUTIONS_DIR / "task1_quaternion.svg", title="Task1 quaternion")


# ----------------------------- task 3 ----------------------------- #


def lemniscate_pos(t: float) -> Tuple[float, float, float]:
    s = math.sin(t)
    c = math.cos(t)
    denom = s * s + 1.0
    x = 10.0 * c / denom
    y = 10.0 * s * c / denom
    z = 10.0
    return x, y, z


def lemniscate_vel(t: float) -> Tuple[float, float, float]:
    s = math.sin(t)
    c = math.cos(t)
    denom = (s * s + 1.0) ** 2
    vx = -10.0 * s * (3.0 - s * s) / denom
    vy = 10.0 * (c * c - 2.0 * s * s) / denom
    vz = 0.0
    return vx, vy, vz


def lemniscate_acc(t: float) -> Tuple[float, float, float]:
    s = math.sin(t)
    c = math.cos(t)
    denom = s * s + 1.0
    D = denom * denom
    dD = 4.0 * s * c * denom

    N1 = -10.0 * s * (3.0 - s * s)
    dN1 = -10.0 * (3.0 * c - 3.0 * s * s * c)
    ax = (dN1 * D - N1 * dD) / (D * D)

    N2 = 10.0 * (c * c - 2.0 * s * s)
    dN2 = -60.0 * s * c
    ay = (dN2 * D - N2 * dD) / (D * D)

    return ax, ay, 0.0


def compute_task3():
    g = 9.81
    dt = 0.02
    t = 0.0
    rows = []
    last_heading = (1.0, 0.0, 0.0)

    while t <= 2 * math.pi + 1e-6:
        vx, vy, vz = lemniscate_vel(t)
        speed_xy = math.hypot(vx, vy)
        if speed_xy < 1e-4:
            heading = last_heading
        else:
            psi = math.atan2(vy, vx)
            heading = (math.cos(psi), math.sin(psi), 0.0)
            last_heading = heading

        ax, ay, az = lemniscate_acc(t)
        F = (ax, ay, az + g)
        b3 = normalize(F)
        h = heading
        b2 = cross(b3, h)
        b2 = normalize(b2)
        if math.sqrt(sum(x * x for x in b2)) < 1e-6:
            b2 = (0.0, 1.0, 0.0)
        b1 = cross(b2, b3)

        R = [
            [b1[0], b2[0], b3[0]],
            [b1[1], b2[1], b3[1]],
            [b1[2], b2[2], b3[2]],
        ]  # body axes in world (body->world)
        q = rot_to_quat(R)
        rows.append((t, *q))
        t = round(t + dt, 10)

    out_path = SOLUTIONS_DIR / "df_quaternion.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "w"])
        for t, qx, qy, qz, qw in rows:
            writer.writerow([f"{t:.2f}", f"{qx:.7f}", f"{qy:.7f}", f"{qz:.7f}", f"{qw:.7f}"])
    create_svg(rows, SOLUTIONS_DIR / "task3_quaternion.svg", title="Task3 quaternion")


# ----------------------------- svg plotting ----------------------------- #


def create_svg(series: List[Tuple[float, float, float, float, float]], path: Path, title: str):
    """Create a very lightweight SVG polyline plot for quaternion components."""
    if not series:
        return
    width, height = 900, 400
    margin = 60
    min_t = series[0][0]
    max_t = series[-1][0]
    min_v = min(min(q[1:]) for q in series)
    max_v = max(max(q[1:]) for q in series)
    if max_v == min_v:
        max_v += 1
        min_v -= 1

    def to_xy(idx: int, val: float) -> Tuple[float, float]:
        x = margin + (series[idx][0] - min_t) / (max_t - min_t) * (width - 2 * margin)
        y = height - margin - (val - min_v) / (max_v - min_v) * (height - 2 * margin)
        return x, y

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    labels = ["qx", "qy", "qz", "qw"]

    lines = []
    # Axes
    lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
    lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{margin}" y2="{margin}" stroke="black"/>')
    lines.append(f'<text x="{width/2:.1f}" y="25" font-size="16" text-anchor="middle">{title}</text>')

    for comp in range(4):
        pts = []
        for i, row in enumerate(series):
            x, y = to_xy(i, row[comp + 1])
            pts.append(f"{x:.2f},{y:.2f}")
        poly = " ".join(pts)
        lines.append(f'<polyline fill="none" stroke="{colors[comp]}" stroke-width="1.5" points="{poly}"/>')
        lines.append(
            f'<text x="{width - margin + 5}" y="{margin + 15 * (comp + 1)}" font-size="12" fill="{colors[comp]}">{labels[comp]}</text>'
        )

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(lines) + "</svg>"
    path.write_text(svg)




def main():
    SOLUTIONS_DIR.mkdir(exist_ok=True)
    compute_task1()
    compute_task3()



if __name__ == "__main__":
    main()
