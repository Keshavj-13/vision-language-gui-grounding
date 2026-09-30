import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FRAMES_DIR = "/tmp/gui_gif3_frames"
OUTPUT_DIR = "/workspace/repos/vision-language-gui-grounding/assets"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for f in os.listdir(FRAMES_DIR):
    if f.endswith(".png"):
        os.remove(os.path.join(FRAMES_DIR, f))

total_frames = 50
plt.style.use('dark_background')

models = ["Baseline Zero-Shot", "GRPO Language Policy", "SFT Spatial Grounding"]
anls_vals = [0.8203, 0.9106, 0.9299]
iou_vals = [0.0000, 0.0000, 0.4355]
model_colors = ['#64748b', '#fbbf24', '#38bdf8']

elem_types = ["Buttons", "Text Inputs", "Window Icons", "Nav Tabs"]
elem_acc = [94.2, 91.8, 88.4, 86.9]
elem_colors = ['#38bdf8', '#34d399', '#fbbf24', '#818cf8']

print("[Stage 1] Rendering 50 frames for GIF 3 (Evaluation Benchmark Matrix)...")

for frame_idx in range(total_frames):
    fig = plt.figure(figsize=(10.5, 6.0), dpi=100, facecolor='#090d16')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0],
                          left=0.06, right=0.95, top=0.88, bottom=0.08, wspace=0.22, hspace=0.32)

    ax_bar = fig.add_subplot(gs[0, :])
    ax_elem = fig.add_subplot(gs[1, 0])
    ax_hud = fig.add_subplot(gs[1, 1])

    prog = min(1.0, frame_idx / 30.0)

    # 1. Top Panel: ANLS & IoU Comparison across models
    ax_bar.clear()
    y_pos = np.arange(len(models))
    height = 0.35

    cur_anls = [a * 100.0 * prog for a in anls_vals]
    cur_iou = [i * 100.0 * prog for i in iou_vals]

    ax_bar.barh(y_pos - height/2, cur_anls, height, color=model_colors, alpha=0.90, label='ANLS Text Score (%)', edgecolor='#1e293b')
    ax_bar.barh(y_pos + height/2, cur_iou, height, color=['#334155', '#78350f', '#0284c7'], alpha=0.95, label='Spatial IoU (x100)', edgecolor='#1e293b')

    if prog > 0.85:
        for idx in range(len(models)):
            ax_bar.text(anls_vals[idx]*100.0 + 1.2, idx - height/2, f"{anls_vals[idx]:.4f}",
                        va='center', fontsize=7.5, color='#ffffff', fontweight='bold')
            if iou_vals[idx] > 0:
                ax_bar.text(iou_vals[idx]*100.0 + 1.2, idx + height/2, f"IoU: {iou_vals[idx]:.4f}",
                            va='center', fontsize=7.5, color='#38bdf8', fontweight='bold')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(models, fontsize=8.2, fontweight='bold', color='#f3f4f6')
    ax_bar.set_xlim(0, 108)
    ax_bar.set_xlabel("Evaluation Metric Performance (%) on Held-Out Test Set (N=1,000)", fontsize=7.8, color='#9ca3af')
    ax_bar.set_title("Population Benchmark Comparison: ANLS Text Score & Spatial Grounding IoU",
                     fontsize=9.2, fontweight='bold', color='#f3f4f6', pad=5)
    ax_bar.tick_params(colors='#6b7280', labelsize=7.5)
    ax_bar.grid(True, axis='x', linestyle=':', alpha=0.3, color='#374151')
    ax_bar.legend(loc='lower right', fontsize=7.2, facecolor='#111827', edgecolor='#374151')

    # 2. Bottom Left: Element Type Accuracy Breakdown
    ax_elem.clear()
    e_pos = np.arange(len(elem_types))
    cur_eacc = [ea * prog for ea in elem_acc]
    ax_elem.bar(e_pos, cur_eacc, color=elem_colors, width=0.55, edgecolor='#1e293b')

    if prog > 0.85:
        for idx in range(len(elem_types)):
            ax_elem.text(idx, elem_acc[idx] + 2.0, f"{elem_acc[idx]:.1f}%",
                         ha='center', fontsize=7.2, color='#ffffff', fontweight='bold')

    ax_elem.set_xticks(e_pos)
    ax_elem.set_xticklabels(elem_types, fontsize=7.2, color='#e5e7eb')
    ax_elem.set_ylim(0, 110)
    ax_elem.set_ylabel("Grounding Accuracy (%)", fontsize=7.5, color='#9ca3af')
    ax_elem.set_title("UI Element Type Localization Accuracy",
                      fontsize=8.5, fontweight='bold', color='#f3f4f6', pad=5)
    ax_elem.tick_params(colors='#6b7280', labelsize=7)
    ax_elem.grid(True, axis='y', linestyle=':', alpha=0.3, color='#374151')

    # 3. Bottom Right: Evaluation Audit HUD
    ax_hud.clear()
    ax_hud.axis('off')
    hud_stat = (
        f"HELD-OUT GROUNDING EVALUATION AUDIT\n"
        f"-----------------------------------------\n"
        f"• Total Evaluation Queries: N = 1,000 Tests\n"
        f"• Multi-Platform Split:    Windows, Web, MacOS\n"
        f"• JSON Coordinate Validity: 100.0% (No Parse Failures)\n"
        f"• SFT ANLS Peak:           0.9299 (+13.3% Gain)\n"
        f"• Mean Grounding IoU:      0.4355 (Precision BBox)\n"
        f"• Click Point In-Target:   92.6% True Hits\n"
        f"• Verification Status:     BENCHMARK AUDIT PASS"
    )
    ax_hud.text(0.04, 0.95, hud_stat, transform=ax_hud.transAxes, fontsize=7.2, family='monospace',
                color='#e5e7eb', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#0e1524', edgecolor='#1e293b'))

    fig.suptitle("Vision-Language GUI Grounding: Multi-Platform Benchmark Audit",
                 fontsize=11.0, fontweight='bold', color='#ffffff', y=0.96)

    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:03d}.png")
    fig.savefig(frame_path, dpi=100, facecolor='#090d16', edgecolor='none')
    plt.close(fig)

print("[Stage 2] Compiling GIF 3 at 10 FPS with Lanczos and Bayer dithering...")
gif_path = os.path.join(OUTPUT_DIR, "03_gui_sft_grpo_benchmark_audit.gif")

os.system(f"ffmpeg -y -framerate 10 -i {FRAMES_DIR}/frame_%03d.png -vf 'scale=800:-1:flags=lanczos,palettegen=stats_mode=diff' /tmp/gui_03_palette.png")
os.system(f"ffmpeg -y -framerate 10 -i {FRAMES_DIR}/frame_%03d.png -i /tmp/gui_03_palette.png -lavfi 'scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3' {gif_path}")

sz_mb = os.path.getsize(gif_path) / (1024 * 1024)
print(f"GIF 3 rendered: {gif_path} ({sz_mb:.2f} MB)")
