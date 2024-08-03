# TATA Steel Summer Internship 2024

## Overview
Welcome to the repository documenting my TATA Steel Summer Internship in 2024. This guide provides a comprehensive overview of my 5-week journey, including week-wise details, the tech stack used, installation instructions, and insights on various large language models (LLMs).

## Table of Contents
1. [Week 1: Introduction to PyTorch](#week-1-introduction-to-pytorch)
2. [Week 2: Basic Physics Laws](#week-2-basic-physics-laws)
3. [Week 3: Implementing Physics-Informed Neural Networks (PINNs)](#week-3-implementing-physics-informed-neural-networks-pinns)
4. [Week 4: Advanced Courses and Certifications](#week-4-advanced-courses-and-certifications)
5. [Week 5: Exploring Large Language Models](#week-5-exploring-large-language-models)
6. [Installation](#installation)
7. [Tech Stack](#tech-stack)
8. [Notes on Large Language Models (LLMs)](#notes-on-large-language-models)

## Week 1: Introduction to PyTorch
- **Objective**: Learn the basics of PyTorch
- **Activities**:
  - Completed PyTorch tutorials
  - Implemented basic neural network models
  - Gained familiarity with tensors, autograd, and model training

## Week 2: Basic Physics Laws
- **Objective**: Study basic physics laws
- **Activities**:
  - Reviewed Burgers, Heat, and Navier-Stokes equations
  - Understood their applications in real-world scenarios
  - Prepared to implement these equations in neural network models

## Week 3: Implementing Physics-Informed Neural Networks (PINNs)
- **Objective**: Implement PINNs using the studied physics laws
- **Activities**:
  - Implemented Burgers, Heat, and Navier-Stokes equations with PINNs
  - Applied the models to a real-world dataset
  - Improved results using the DeepXDE library

## Week 4: Advanced Courses and Certifications
- **Objective**: Gain advanced knowledge and certifications
- **Activities**:
  - Completed courses on fine-tuning LLMs and ChatGPT prompt engineering from DeepLearning.AI
  - Completed a course on PINNs from Udemy and earned a certificate
  - Studied the DeepSDE library and its applications

## Week 5: Exploring Large Language Models
- **Objective**: Explore different types of LLMs and their uses
- **Activities**:
  - Explored various LLMs, their pros and cons
  - Applied generative AI to improve PINNs with the TATA dataset
  - Achieved a 30% improvement in accuracy through over 10,000 simulations

## Installation
To replicate the work done during the internship, you'll need to install the following libraries:

### PyTorch
```bash
pip install torch torchvision torchaudio
Installation
Step 1: Clone the Repository
First, clone this repository to your local machine using the following command:

git clone https://github.com/your-username/TATA-STEEL-Summer-Intern.git
cd TATA-STEEL-Summer-Intern

Step 2: Set Up a Virtual Environment
It is recommended to create a virtual environment to manage your dependencies. You can create a virtual environment using venv:
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install torch torchvision torchaudio
pip install deepxde
pip install numpy pandas matplotlib seaborn
import torch
import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")

## Notes on Large Language Models (LLMs)

During the internship, I explored various LLMs to understand their capabilities and applications. Here are some key insights:

### GPT-3
- **Developer**: OpenAI
- **Description**: A state-of-the-art language model known for its ability to understand and generate human-like text.
- **Applications**: Chatbots, content creation, translation, summarization, and more.
- **Strengths**: Versatile, highly accurate, and capable of generating coherent and contextually relevant responses.

### BERT
- **Developer**: Google
- **Description**: Bidirectional Encoder Representations from Transformers, designed to understand the context of words in a sentence.
- **Applications**: Question answering, language translation, text classification.
- **Strengths**: Excellent at understanding the nuances of language, especially for tasks requiring deep understanding of text.

### T5
- **Developer**: Google
- **Description**: Text-to-Text Transfer Transformer, treats every NLP task as a text-to-text problem.
- **Applications**: Translation, summarization, question answering, and more.
- **Strengths**: Highly versatile and powerful for various NLP tasks, consistent performance across different applications.

### RoBERTa
- **Developer**: Facebook AI
- **Description**: A robustly optimized version of BERT, designed to improve upon BERT’s pretraining methodology.
- **Applications**: Same as BERT (question answering, text classification, etc.).
- **Strengths**: Improved performance on many NLP benchmarks, robust understanding of language context.

### LLaMA
- **Developer**: Meta (Facebook AI)
- **Description**: Large Language Model Meta AI, designed to push the boundaries of what LLMs can achieve.
- **Applications**: Broad range of NLP tasks including text generation, summarization, and more.
- **Strengths**: Optimized for performance and scalability, advanced language understanding.

### Cohere
- **Developer**: Cohere AI
- **Description**: Provides a range of language models designed for different NLP tasks, with a focus on ease of use and integration.
- **Applications**: Text classification, generation, summarization, and more.
- **Strengths**: User-friendly API, strong performance on a variety of tasks, customizable models.

### Mistral
- **Developer**: Mistral AI
- **Description**: Focuses on developing advanced language models with high accuracy and performance.
- **Applications**: Various NLP tasks, with an emphasis on precise and efficient text understanding.
- **Strengths**: High accuracy, efficient training and inference, tailored for specific use cases.

### DeepSDE
- **Developer**: Community-driven (various contributors)
- **Description**: A library for stochastic differential equations, used for improving Physics-Informed Neural Networks (PINNs).
- **Applications**: Enhancing model accuracy in scenarios with low data availability, simulations involving complex physical processes.
- **Strengths**: Integrates well with PINNs, improves model performance through extensive simulations.

### OpenAI's LLM Suite
- **Developer**: OpenAI
- **Description**: Includes models like GPT-3, Codex, and others designed for various NLP and coding tasks.
- **Applications**: Text generation, code completion, translation, summarization, and more.
- **Strengths**: Versatility, high accuracy, strong contextual understanding, widely used and supported.

