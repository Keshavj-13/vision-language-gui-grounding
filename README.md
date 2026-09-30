# Vision-Language GUI Grounding: SFT & GRPO Spatial Policy

[![VLM](https://img.shields.io/badge/VLM-Spatial_Grounding-blue?style=for-the-badge)](#)
[![ANLS](https://img.shields.io/badge/ANLS-0.9299_(+13.3%25)-success?style=for-the-badge)](#)
[![IoU](https://img.shields.io/badge/Grounding_IoU-0.4355-success?style=for-the-badge)](#)

A Vision-Language model spatial grounding framework fine-tuned with Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) for autonomous UI element detection and document question answering.

---

## 1. Grounding Progression

| Model Stage | Average Normalized Levenshtein (ANLS) | Grounding IoU | JSON Format Validity |
| :--- | :---: | :---: | :---: |
| **Baseline VLM** | 0.8203 | 0.0000 | 99.7% |
| **GRPO Grounding Policy** | 0.9106 | 0.2810 | 100.0% |
| **SFT + GRPO Merged** | **0.9299** | **0.4355** | **100.0%** |
