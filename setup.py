# ABOUTME: Package configuration for the Reka Edge vLLM plugin.
# ABOUTME: Registers Reka models, tokenizer, and configs with vLLM via entry_points.
from setuptools import find_packages, setup

setup(
    name='vllm-reka',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'tiktoken',
        'regex',
        'opencv-python',
        'numpy',
        'Pillow',
    ],
    entry_points={
        'vllm.general_plugins': [
            'register_reka = vllm_reka:register',
        ]
    },
)
