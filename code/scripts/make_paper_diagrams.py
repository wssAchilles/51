from __future__ import annotations

from pathlib import Path
import sys
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
sys.path.insert(0, str(ROOT / "src"))

from slope_warning.common.plotting import configure_chinese_fonts


def setup_axes(figsize: tuple[float, float] = (10, 5.625)):
    configure_chinese_fonts()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, wh, title, subtitle="", face="#F6FAFF", edge="#2F5C8F", title_size=13, sub_size=10.5):
    x, y = xy
    w, h = wh
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, weight="bold", color="#17324D", zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=sub_size, color="#35536A", linespacing=1.25, zorder=4)
    return patch


def arrow(ax, p1, p2, color="#3A5F7D", lw=1.8, rad=0.0):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=1,
    )


def small_curve(ax, x0, y0, w, h, color="#2F75B5", style="mono"):
    t = np.linspace(0, 1, 120)
    if style == "jump":
        y = 0.18 + 0.18 * t + 0.35 / (1 + np.exp(-16 * (t - 0.62)))
    elif style == "stage":
        y = np.piecewise(t, [t < 0.45, (t >= 0.45) & (t < 0.75), t >= 0.75], [lambda z: 0.16 + 0.18 * z, lambda z: 0.24 + 0.55 * (z - 0.45), lambda z: 0.41 + 1.10 * (z - 0.75)])
    elif style == "residual":
        y = 0.45 + 0.08 * np.sin(18 * t) + 0.03 * np.cos(37 * t)
    else:
        y = 0.12 + 0.78 * t**1.45
    ax.plot(x0 + w * t, y0 + h * y, color=color, lw=2.0)
    ax.plot([x0, x0, x0 + w], [y0 + h, y0, y0], color="#BFC7CF", lw=0.9)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first_float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return float(value)
    raise ValueError(f"no numeric value found in {keys}")


def figure_overall_route():
    fig, ax = setup_axes()
    ax.text(0.5, 0.95, "鲁棒校准—复合证据分段—残差预测—速度预警的整体建模链", ha="center", va="center", fontsize=18, weight="bold", color="#16324F")
    ax.text(0.5, 0.905, "以阶段规律为主线，将五个问题组织为同一条可复核的边坡变形证据链", ha="center", va="center", fontsize=12, color="#4B647A")

    xs = [0.05, 0.24, 0.43, 0.62, 0.81]
    titles = ["问题一\n量测校准", "问题二\n阶段识别", "问题三\n异常补齐", "问题四\n位移预测", "问题五\n预警闭环"]
    subs = [
        "Huber稳健仿射\nBootstrap区间",
        "位移趋势+速度跃迁\n持续加速度证据",
        "状态空间补齐\n变量专属阈值",
        "阶段归一化基线\n扰动残差收缩",
        "五变量组合验证\n逆速度提前量",
    ]
    colors = ["#EAF3FF", "#EAF8F0", "#FFF6E5", "#F1F0FF", "#FFEFEF"]
    edges = ["#2F75B5", "#37966F", "#C98B2B", "#6F65B8", "#B84D4D"]
    for i, x in enumerate(xs):
        box(ax, (x, 0.62), (0.14, 0.18), titles[i], subs[i], colors[i], edges[i], title_size=12)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.145, 0.71), (xs[i + 1] - 0.006, 0.71))

    for i, x in enumerate(xs):
        small_curve(ax, x + 0.018, 0.31, 0.105, 0.18, edges[i], ["mono", "stage", "residual", "jump", "stage"][i])
        ax.text(x + 0.07, 0.25, ["校准残差收敛", "三阶段单调演化", "异常点可解释", "预测曲线单调", "阈值逐级触发"][i], ha="center", fontsize=10.5, color="#34495E")

    box(ax, (0.09, 0.08), (0.82, 0.11), "证据链闭环", "题面输出表格  →  模型对照  →  敏感性分析  →  工程解释  →  可复现代码", "#F7F9FB", "#6B7C8F", title_size=13)
    save(fig, "paper_overall_model_route.png")


def figure_problem_relationship():
    fig, ax = setup_axes((10.5, 5.6))
    ax.text(0.5, 0.95, "五个问题的逻辑关系：由量测统一到工程预警", ha="center", va="center", fontsize=18, weight="bold", color="#16324F")
    ax.text(0.5, 0.905, "每一问的输出都进入后续问题，形成校准、分段、清洗、预测和预警的递进证据链", ha="center", va="center", fontsize=12, color="#4B647A")

    steps = [
        ("数据校准", "问题一\n统一位移量测基准", "#EAF3FF", "#2F75B5"),
        ("阶段识别", "问题二\n确定变形阶段节点", "#EAF8F0", "#37966F"),
        ("异常处理\n变量贡献", "问题三\n生成可信扰动变量", "#FFF6E5", "#C98B2B"),
        ("阶段预测", "问题四\n阶段归一化基线+残差", "#F1F0FF", "#6F65B8"),
        ("变量筛选\n预警机制", "问题五\n速度阈值+逆速度", "#FFEFEF", "#B84D4D"),
    ]
    xs = [0.05, 0.24, 0.43, 0.62, 0.81]
    for i, (title, sub, face, edge) in enumerate(steps):
        box(ax, (xs[i], 0.57), (0.14, 0.20), title, sub, face, edge, title_size=12.5, sub_size=9.7)
        if i < 4:
            arrow(ax, (xs[i] + 0.145, 0.67), (xs[i + 1] - 0.006, 0.67), "#3A5F7D")

    notes = [
        ("消除传感器\n尺度差异", xs[0]),
        ("提供阶段标签\n和速度基准", xs[1]),
        ("提供补齐样本\n与主控变量", xs[2]),
        ("输出预测曲线\n与风险区间", xs[3]),
        ("形成工程\n处置闭环", xs[4]),
    ]
    for text, x in notes:
        ax.text(x + 0.07, 0.455, text, ha="center", va="center", fontsize=9.5, color="#34495E", linespacing=1.20)
        arrow(ax, (x + 0.07, 0.57), (x + 0.07, 0.49), "#9AA7B2", lw=1.2)

    box(ax, (0.13, 0.17), (0.74, 0.16), "贯穿全文的共同约束", "时间块验证避免未来信息泄漏；单调位移约束保证工程含义\n多模型对照与敏感性分析形成稳健性证据", "#F7F9FB", "#6B7C8F", title_size=13, sub_size=9.8)
    arrow(ax, (0.69, 0.32), (0.31, 0.57), "#6B7C8F", rad=-0.23)
    arrow(ax, (0.86, 0.57), (0.86, 0.32), "#B84D4D", rad=0.0)
    save(fig, "paper_problem_relationship.png")


def figure_preprocess():
    fig, ax = setup_axes((8.8, 7.0))
    ax.text(0.5, 0.955, "多源监测数据预处理与特征构造框架", ha="center", va="center", fontsize=17, weight="bold", color="#16324F")
    ax.text(0.5, 0.915, "先区分变量物理属性，再选择补齐、异常识别与后向窗口特征", ha="center", va="center", fontsize=11.5, color="#4B647A")

    top_y = 0.71
    top_w, top_h = 0.18, 0.16
    top = [
        ("原始数据", "位移 / 降雨 / 孔压\n微震 / 爆破", "#EAF3FF", "#2F75B5", 0.05),
        ("时间统一", "时间戳转换\n10 min步长", "#EAF8F0", "#37966F", 0.29),
        ("变量分类", "连续量 / 稀疏量\n事件量", "#FFF6E5", "#C98B2B", 0.53),
        ("质量复核", "缺失 / 非负\n单调 / 量纲", "#FFEFEF", "#B84D4D", 0.77),
    ]
    for title, sub, face, edge, x in top:
        box(ax, (x, top_y), (top_w, top_h), title, sub, face, edge, title_size=11.8, sub_size=9.3)
    for x1, x2 in [(0.23, 0.29), (0.47, 0.53), (0.71, 0.77)]:
        arrow(ax, (x1, top_y + top_h / 2), (x2, top_y + top_h / 2))

    ax.text(0.5, 0.635, "按变量机制分流处理", ha="center", va="center", fontsize=12.5, weight="bold", color="#263A4A")
    branch_y = 0.42
    branch_w, branch_h = 0.24, 0.17
    branch = [
        ("连续状态量", "Kalman平滑\n残差阈值", "#EAF3FF", "#2F75B5", 0.08),
        ("稀疏驱动量", "保留真实峰值\n分位阈值", "#EAF8F0", "#37966F", 0.38),
        ("爆破事件量", "空值=无爆破\n冲击衰减项", "#FFF6E5", "#C98B2B", 0.68),
    ]
    source = (0.53 + top_w / 2, top_y)
    for title, sub, face, edge, x in branch:
        box(ax, (x, branch_y), (branch_w, branch_h), title, sub, face, edge, title_size=11.8, sub_size=9.6)
        arrow(ax, source, (x + branch_w / 2, branch_y + branch_h), edge, rad=0.04 if x < 0.38 else -0.04)

    box(ax, (0.17, 0.17), (0.28, 0.15), "后向窗口特征", "只用历史信息\n累积量与变化率", "#F1F0FF", "#6F65B8", title_size=11.8, sub_size=9.6)
    box(ax, (0.55, 0.17), (0.28, 0.15), "时间块验证", "连续分块CV\n避免未来泄漏", "#F7F9FB", "#6B7C8F", title_size=11.8, sub_size=9.6)
    arrow(ax, (0.20, branch_y), (0.31, 0.33), "#2F75B5")
    arrow(ax, (0.50, branch_y), (0.31, 0.33), "#37966F")
    arrow(ax, (0.80, branch_y), (0.69, 0.33), "#C98B2B")
    arrow(ax, (0.45, 0.245), (0.55, 0.245))
    ax.text(0.5, 0.085, "输出：无缺失建模样本、变量专属异常标记、无未来泄漏的时序特征", ha="center", fontsize=11.0, color="#263A4A")
    save(fig, "paper_preprocess_framework.png")


def figure_q2_composite():
    fig, ax = setup_axes()
    ax.text(0.5, 0.95, "问题二复合证据阶段识别：区分阶段起点与速度显著跃迁点", ha="center", va="center", fontsize=18, weight="bold", color="#16324F")
    ax.text(0.5, 0.905, "累计位移给出趋势开始改变，持续加速度提供提前证据，速度跃迁确认阶段状态改变", ha="center", va="center", fontsize=12, color="#4B647A")

    panels = [("位移趋势变化", "分段拟合误差下降\n通常偏早", 0.06, "#2F75B5", "stage"), ("持续加速度起点", "连续窗口速度抬升\n具备预警意义", 0.38, "#37966F", "jump"), ("速度水平跃迁", "速度均值显著改变\n强证据", 0.70, "#C98B2B", "jump")]
    for title, sub, x, color, style in panels:
        box(ax, (x, 0.58), (0.24, 0.22), title, sub, "#F7FAFC", color)
        small_curve(ax, x + 0.04, 0.38, 0.16, 0.13, color, style)
        arrow(ax, (x + 0.12, 0.38), (0.50, 0.29), color, rad=0.1 if x < 0.4 else -0.1)

    box(ax, (0.32, 0.16), (0.36, 0.15), "加权中位融合得到阶段主节点", "权重体现证据可靠性：速度跃迁 > 持续加速度 > 位移趋势", "#FFEFEF", "#B84D4D")
    ax.text(0.50, 0.08, r"$T=\operatorname{wmed}\{(c_j,\omega_j)\}$，并用阶段内拟合误差和平均速度单调性复核", ha="center", fontsize=12.5, color="#263A4A")
    save(fig, "q2_composite_evidence_fusion.png")


def figure_q4_model():
    fig, ax = setup_axes()
    ax.text(0.5, 0.95, "问题四阶段归一化基线与扰动残差预测结构", ha="center", va="center", fontsize=18, weight="bold", color="#16324F")
    ax.text(0.5, 0.905, "主形变由阶段单调基线承担，机器学习只修正扰动残差，从结构上保证预测曲线整体单调", ha="center", va="center", fontsize=12, color="#4B647A")

    box(ax, (0.05, 0.64), (0.18, 0.15), "阶段标签", "缓慢 / 加速 / 快速", "#EAF3FF", "#2F75B5")
    box(ax, (0.28, 0.64), (0.18, 0.15), "归一化时间", r"$\tau=(t-t_{k,0})/(t_{k,1}-t_{k,0})$", "#EAF8F0", "#37966F", title_size=12)
    box(ax, (0.51, 0.64), (0.18, 0.15), "单调PCHIP基线", "阶段内形变骨架\n累积最大投影", "#FFF6E5", "#C98B2B")
    box(ax, (0.76, 0.64), (0.18, 0.15), "实验集预测", "基线 + 收缩残差\n95%预测区间", "#FFEFEF", "#B84D4D")
    arrow(ax, (0.23, 0.715), (0.28, 0.715))
    arrow(ax, (0.46, 0.715), (0.51, 0.715))
    arrow(ax, (0.69, 0.715), (0.76, 0.715))

    ax.text(0.16, 0.49, "阶段单调基线", ha="center", fontsize=12, weight="bold", color="#2F75B5")
    for i, color in enumerate(["#2F75B5", "#37966F", "#B84D4D"]):
        t = np.linspace(0, 1, 80)
        x0 = 0.06 + i * 0.10
        ax.plot(x0 + 0.08 * t, 0.30 + 0.16 * (t ** (1.2 + 0.4 * i)) + i * 0.025, lw=2.2, color=color)
    ax.plot([0.05, 0.35], [0.28, 0.28], color="#BFC7CF", lw=0.9)
    ax.plot([0.05, 0.05], [0.28, 0.50], color="#BFC7CF", lw=0.9)

    box(ax, (0.42, 0.32), (0.22, 0.16), "扰动残差模型", "降雨、孔压、微震\n爆破冲击衰减项", "#F1F0FF", "#6F65B8")
    ax.text(0.53, 0.24, r"$\hat s_t=B_{k,0}+g_k(\tau_t)+0.35\,f_k(X_t)$", ha="center", fontsize=13, color="#263A4A")
    arrow(ax, (0.35, 0.39), (0.42, 0.40))
    arrow(ax, (0.64, 0.40), (0.78, 0.39))

    small_curve(ax, 0.75, 0.30, 0.17, 0.16, "#B84D4D", "mono")
    ax.fill_between([0.75, 0.92], [0.38, 0.48], [0.34, 0.44], color="#B84D4D", alpha=0.12)
    save(fig, "q4_stage_residual_model.png")


def figure_q5_warning():
    fig, ax = setup_axes()
    ax.text(0.5, 0.95, "问题五三级速度预警与逆速度提前量验证闭环", ha="center", va="center", fontsize=18, weight="bold", color="#16324F")
    ax.text(0.5, 0.905, "预警触发不是单次超阈值，而是速度、持续时间、多源扰动和逆速度趋势共同确认", ha="center", va="center", fontsize=12, color="#4B647A")

    nodes = {
        "data": ((0.08, 0.62), "实时监测数据", "位移 + 五类扰动变量", "#EAF3FF", "#2F75B5"),
        "speed": ((0.36, 0.72), "速度清洗与平滑", "Hampel + 6 h滚动中位", "#EAF8F0", "#37966F"),
        "level": ((0.66, 0.62), "三级阈值判别", "关注 / 预警 / 严重预警", "#FFF6E5", "#C98B2B"),
        "inverse": ((0.63, 0.34), "逆速度趋势", r"$1/\tilde v_t=a+bt,\ b<0$", "#F1F0FF", "#6F65B8"),
        "action": ((0.34, 0.22), "工程处置反馈", "巡检、加密监测、风险复核", "#FFEFEF", "#B84D4D"),
        "disturb": ((0.08, 0.34), "扰动增强确认", "降雨 / 孔压 / 微震 / 爆破", "#F7F9FB", "#6B7C8F"),
    }
    centers = {}
    for key, (xy, title, sub, face, edge) in nodes.items():
        box(ax, xy, (0.22, 0.14), title, sub, face, edge)
        centers[key] = (xy[0] + 0.11, xy[1] + 0.07)
    arrow(ax, centers["data"], centers["speed"], "#2F75B5")
    arrow(ax, centers["speed"], centers["level"], "#37966F")
    arrow(ax, centers["level"], centers["inverse"], "#C98B2B")
    arrow(ax, centers["inverse"], centers["action"], "#6F65B8")
    arrow(ax, centers["action"], centers["disturb"], "#B84D4D")
    arrow(ax, centers["disturb"], centers["data"], "#6B7C8F")
    arrow(ax, centers["disturb"], centers["level"], "#6B7C8F", rad=0.15)

    ax.text(0.50, 0.52, "触发条件", ha="center", fontsize=13, weight="bold", color="#263A4A", zorder=5)
    ax.text(0.50, 0.46, "速度超阈值  +  持续时间  +  扰动增强  +  逆速度下降", ha="center", fontsize=12.5, color="#263A4A", zorder=5)
    ax.plot([0.38, 0.62], [0.43, 0.43], color="#BFC7CF", lw=1.1, zorder=1)
    save(fig, "q5_warning_closed_loop.png")


def figure_q3_contribution_bar():
    configure_chinese_fonts()
    perm_rows = read_csv_rows(ROOT / "outputs" / "tables" / "q3_grouped_contribution_permutation.csv")
    stable_rows = read_csv_rows(ROOT / "outputs" / "award" / "q3_variable_contribution_stability.csv")
    name_map = {"pore": "孔隙水压力", "deep": "深部位移", "rain": "降雨量", "micro": "微震事件数"}
    stable = {row["factor"]: (float(row["mean"]), float(row["std"])) for row in stable_rows}
    rows = []
    for row in perm_rows:
        key = row["factor"]
        rows.append((name_map.get(key, key), float(row["permutation_RMSE_increase"]), stable[key][0], stable[key][1]))
    rows.sort(key=lambda x: x[1])

    labels = [r[0] for r in rows]
    perm = np.array([r[1] for r in rows])
    mean = np.array([r[2] for r in rows])
    std = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    fig.patch.set_facecolor("white")
    y = np.arange(len(labels))
    colors = ["#87A9D8", "#78B79A", "#E3B35A", "#C97878"]
    ax.barh(y, perm, color=colors[: len(labels)], alpha=0.88, label="置换RMSE增量")
    ax.errorbar(mean, y, xerr=std, fmt="o", color="#263A4A", capsize=4, label="重复划分均值±标准差")
    label_x = np.maximum(perm + 0.18, mean + std + 0.24)
    ax.set_xlim(0, float(label_x.max() + 0.55))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("贡献度 / RMSE增量 (mm)", fontsize=11)
    ax.set_title("问题三变量贡献稳定性对照", fontsize=16, weight="bold", color="#16324F")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right", frameon=False, fontsize=10, borderaxespad=0.8)
    for i, value in enumerate(perm):
        ax.text(label_x[i], i, f"{value:.2f}", va="center", ha="left", fontsize=10, color="#263A4A")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.86, bottom=0.16)
    save(fig, "q3_variable_contribution_bar.png")


def figure_model_sensitivity_summary():
    configure_chinese_fonts()
    q2_rows = read_csv_rows(ROOT / "outputs" / "award" / "q2_transition_sensitivity.csv")
    q5_rows = read_csv_rows(ROOT / "outputs" / "tables" / "q5_warning_window_sensitivity.csv")

    velocity_rows = [r for r in q2_rows if r["method"] == "velocity_level"]
    windows = [f"{float(r['window_h']):.0f}h" for r in velocity_rows]
    break1 = [float(r["break1_serial"]) for r in velocity_rows]
    break2 = [float(r["break2_serial"]) for r in velocity_rows]

    q5 = [r for r in q5_rows if r["window"] in {"3h", "6h", "12h"} and r["stage"] in {"1", "2", "3"}]
    q5_windows = ["3h", "6h", "12h"]
    severe = []
    focus = []
    for w in q5_windows:
        stage1 = next(r for r in q5 if r["window"] == w and r["stage"] == "1")
        stage3 = next(r for r in q5 if r["window"] == w and r["stage"] == "3")
        focus.append(first_float(stage1, "q90"))
        severe.append(first_float(stage3, "q50", "q75", "q90"))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    fig.patch.set_facecolor("white")
    ax = axes[0]
    x = np.arange(len(windows))
    ax.plot(x, break1, marker="o", lw=2.2, color="#2F75B5", label="第一速度跃迁点")
    ax.plot(x, break2, marker="o", lw=2.2, color="#B84D4D", label="第二速度跃迁点")
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("节点编号")
    ax.set_title("阶段识别窗口敏感性", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(len(q5_windows))
    width = 0.34
    ax.bar(x - width / 2, focus, width, color="#E3B35A", label="关注阈值候选")
    ax.bar(x + width / 2, severe, width, color="#B84D4D", label="严重预警候选")
    ax.set_xticks(x)
    ax.set_xticklabels(q5_windows)
    ax.set_ylabel("速度 (mm/h)")
    ax.set_title("预警阈值窗口敏感性", fontsize=13, weight="bold")
    ax.set_ylim(0, max(severe) * 1.12)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.72), borderaxespad=0.0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("模型关键参数敏感性汇总", fontsize=16, weight="bold", color="#16324F")
    fig.tight_layout(rect=(0, 0, 0.88, 0.92))
    save(fig, "model_sensitivity_summary.png")


def main() -> None:
    figure_overall_route()
    figure_problem_relationship()
    figure_preprocess()
    figure_q2_composite()
    figure_q4_model()
    figure_q5_warning()
    figure_q3_contribution_bar()
    figure_model_sensitivity_summary()


if __name__ == "__main__":
    main()
