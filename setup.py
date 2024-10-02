from setuptools import setup, find_packages

setup(
    name='llm_training_pipeline',
    version='0.1.0',
    description='A pipeline for training, evaluating, and deploying GPT-2 models using Ray and PyTorch.',
    author='Glen',
    author_email='glenbp20@gmail.com',
    packages=find_packages(exclude=['tests', 'notebooks']),
    install_requires=[
        'torch==1.13.1',
        'transformers==4.24.0',
        'ray[default]==2.0.0',
        'mlflow==1.29.0',
        'pyyaml==6.0',
        'tqdm==4.64.1',
        'fastapi==0.85.1',
        'uvicorn==0.18.3',
        'wandb==0.13.4',
        'jsonschema==4.16.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
