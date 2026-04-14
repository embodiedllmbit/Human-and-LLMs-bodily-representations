#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model configuration definitions for the released final-target experiment bundle."""

from pathlib import Path

EXPERIMENT_ROOT = str(Path(__file__).resolve().parent)
VIDEO_DIR = f"{EXPERIMENT_ROOT}/experiment_videos"

MODELS = {
    "InternVL3.5": {
        "script_path": f"{EXPERIMENT_ROOT}/InternVL3.5/InternVL3.5-chat.py",
        "supports_think_tag": True,
        "is_thinking_model": False,
        "modes": ["base", "think"],
        "mode_arg_name": "mode",
        "mode_arg_values": {"base": "base", "think": "thinking"},
    },
    "GLM-4.1V-base": {
        "script_path": f"{EXPERIMENT_ROOT}/GLM-4.1V-base/GLM-4.1V-base-chat.py",
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "GLM-4.1V-thinking": {
        "script_path": f"{EXPERIMENT_ROOT}/GLM-4.1V-thinking/GLM-4.1V-thinking-chat.py",
        "supports_think_tag": False,
        "is_thinking_model": True,
        "modes": ["think"],
    },
    "Qwen": {
        "script_path": f"{EXPERIMENT_ROOT}/Qwen/Qwen-chat.py",
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "Qwen-Thinking": {
        "script_path": f"{EXPERIMENT_ROOT}/Qwen-Thinking/Qwen-Thinking-chat.py",
        "supports_think_tag": False,
        "is_thinking_model": True,
        "modes": ["think"],
    },
    "RynnBrain-8B": {
        "script_path": f"{EXPERIMENT_ROOT}/RynnBrain/RynnBrain-8B-chat.py",
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "RynnBrain-CoP": {
        "script_path": f"{EXPERIMENT_ROOT}/RynnBrain-CoP/RynnBrain-CoP-chat.py",
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "RoboBrain2.5": {
        "script_path": f"{EXPERIMENT_ROOT}/RoboBrain2.5/RoboBrain2.5-chat.py",
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "MiMo-Embodied": {
        "script_path": f"{EXPERIMENT_ROOT}/MiMo-Embodied/MiMo-Embodied-chat.py",
        "supports_think_tag": True,
        "is_thinking_model": False,
        "modes": ["base", "think"],
        "mode_arg_name": "mode",
        "mode_arg_values": {"base": "base", "think": "thinking"},
    },
}


def get_experiment_configs():
    configs = []
    for model_name, config in MODELS.items():
        for mode in config["modes"]:
            configs.append((model_name, mode, config))
    return configs


if __name__ == "__main__":
    for model_name, mode, _ in get_experiment_configs():
        print(f"{model_name}	{mode}")
