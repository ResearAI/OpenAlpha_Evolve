# OpenAlpha_Evolve: Contribute to Improve this Project

![openalpha_evolve_workflow](https://github.com/user-attachments/assets/9d4709ad-0072-44ae-bbb5-7eea1c5fa08c)

OpenAlpha_Evolve is an open-source Python framework inspired by the groundbreaking research on autonomous coding agents like DeepMind's AlphaEvolve. It's a **regeneration** of the core idea: an intelligent system that iteratively writes, tests, and improves code using Large Language Models (LLMs) via LiteLLM, guided by the principles of evolution.

Our mission is to provide an accessible, understandable, and extensible platform for researchers, developers, and enthusiasts to explore the fascinating intersection of AI, code generation, and automated problem-solving.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

## Table of Contents
- [✨ The Vision: AI-Driven Algorithmic Innovation](#-the-vision-ai-driven-algorithmic-innovation)
- [🧠 How It Works: The Evolutionary Cycle](#-how-it-works-the-evolutionary-cycle)
- [🚀 Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [🏁 Getting Started](#-getting-started)
- [💡 Defining Your Own Algorithmic Quests!](#-defining-your-own-algorithmic-quests)
- [🔮 The Horizon: Future Evolution](#-the-horizon-future-evolution)
- [🤝 Join the Evolution: Contributing](#-join-the-evolution-contributing)
- [📜 License](#-license)
- [🙏 Homage](#-homage)

---
![image](https://github.com/user-attachments/assets/ff498bb7-5608-46ca-9357-fd9b55b76800)
![image](https://github.com/user-attachments/assets/c1b4184a-f5d5-43fd-8f50-3e729c104e11)



## ✨ The Vision: AI-Driven Algorithmic Innovation

Imagine an agent that can:

*   Understand a complex problem description.
*   Generate initial algorithmic solutions.
*   Rigorously test its own code.
*   Learn from failures and successes.
*   Evolve increasingly sophisticated and efficient algorithms over time.

OpenAlpha_Evolve is a step towards this vision. It's not just about generating code; it's about creating a system that *discovers* and *refines* solutions autonomously.

---
<img width="1253" alt="Screenshot 2025-05-19 at 12 17 58 AM" src="https://github.com/user-attachments/assets/43d7c5a8-f361-438c-ac38-39717f28ee1f" />

## 🧠 How It Works: The Evolutionary Cycle

OpenAlpha_Evolve employs a modular, agent-based architecture to orchestrate an evolutionary process:

1.  **Task Definition**: You, the user, define the algorithmic "quest" – the problem to be solved, including examples of inputs and expected outputs.
2.  **Prompt Engineering (`PromptDesignerAgent`)**: This agent crafts intelligent prompts for the LLM. It designs:
    *   *Initial Prompts*: To generate the first set of candidate solutions.
    *   *Mutation Prompts*: To introduce variations and improvements to existing solutions, often requesting changes in a "diff" format.
    *   *Bug-Fix Prompts*: To guide the LLM in correcting errors from previous attempts, also typically expecting a "diff".
3.  **Code Generation (`CodeGeneratorAgent`)**: Powered by an LLM (currently configured for Gemini), this agent takes the prompts and generates Python code. If a "diff" is requested and received, it attempts to apply the changes to the parent code.
4.  **Evaluation (`EvaluatorAgent`)**: The generated code is put to the test!
    *   *Syntax Check*: Is the code valid Python?
    *   *Execution*: The code is run in a temporary, isolated environment against the input/output examples defined in the task.
    *   *Fitness Scoring*: Programs are scored based on correctness (how many test cases pass), efficiency (runtime), and other potential metrics.
5.  **Database (`DatabaseAgent`)**: All programs (code, fitness scores, generation, lineage) are stored, creating a record of the evolutionary history (currently in-memory).
6.  **Selection (`SelectionControllerAgent`)**: The "survival of the fittest" principle in action. This agent selects:
    *   *Parents*: Promising programs from the current generation to produce offspring.
    *   *Survivors*: The best programs from both the current population and new offspring to advance to the next generation.
7.  **Iteration**: This cycle repeats for a defined number of generations, with each new generation aiming to produce better solutions than the last.
8.  **Orchestration (`TaskManagerAgent`)**: The maestro of the operation, coordinating all other agents and managing the overall evolutionary loop.

---

## 🚀 Key Features

*   **LLM-Powered Code Generation**: Leverages state-of-the-art Large Language Models via LiteLLM, supporting multiple providers (OpenAI, Anthropic, Google, etc.).
*   **Evolutionary Algorithm Core**: Implements iterative improvement through selection, LLM-driven mutation/bug-fixing using diffs, and survival.
*   **Modular Agent Architecture**: Easily extend or replace individual components (e.g., use a different LLM, database, or evaluation strategy).
*   **Automated Program Evaluation**: Syntax checking and functional testing against user-provided examples. Code execution is sandboxed using **Docker containers** for improved security and dependency management, with configurable timeout mechanisms.
*   **Configuration Management**: Easily tweak parameters like population size, number of generations, LLM models, API settings, and Docker configurations via `config/settings.py` and `.env`.
*   **Detailed Logging**: Comprehensive logs provide insights into each step of the evolutionary process.
*   **Diff-based Mutations**: The system is designed to use diffs for mutations and bug fixes, allowing for more targeted code modifications by the LLM.
*   **Open Source & Extensible**: Built with Python, designed for experimentation and community contributions.

---

## 📂 Project Structure

```text
./
├── agents/                  # Contains the core intelligent agents responsible for different parts of the evolutionary process. Each agent is in its own subdirectory.
│   ├── code_generator/      # Agent responsible for generating code using LLMs.
│   ├── database_agent/      # Agent for managing the storage and retrieval of programs and their metadata.
│   ├── evaluator_agent/     # Agent that evaluates the generated code for syntax, execution, and fitness.
│   ├── prompt_designer/     # Agent that crafts prompts for the LLM for initial generation, mutation, and bug fixing.
│   ├── selection_controller/  # Agent that implements the selection strategy for parent and survivor programs.
│   ├── task_manager/        # Agent that orchestrates the overall evolutionary loop and coordinates other agents.
│   ├── rl_finetuner/        # Placeholder for a future Reinforcement Learning Fine-Tuner agent to optimize prompts.
│   └── monitoring_agent/    # Placeholder for a future Monitoring Agent to track and visualize the process.
├── config/                  # Holds configuration files, primarily `settings.py` for system parameters and API keys.
├── core/                    # Defines core data structures and interfaces, like `Program` and `TaskDefinition`.
├── utils/                   # Contains utility functions and helper classes used across the project (currently minimal).
├── tests/                   # Includes unit and integration tests to ensure code quality and correctness (placeholders, to be expanded).
├── scripts/                 # Stores helper scripts for various tasks, such as generating diagrams or reports.
├── main.py                  # The main entry point to run the OpenAlpha_Evolve system and start an evolutionary run.
├── requirements.txt         # Lists all Python package dependencies required to run the project.
├── .env.example             # An example file showing the environment variables needed, such as API keys. Copy this to `.env` and fill in your values.
├── .gitignore               # Specifies intentionally untracked files that Git should ignore (e.g., `.env`, `__pycache__/`).
├── LICENSE.md               # Contains the full text of the MIT License under which the project is distributed.
└── README.md                # This file! Provides an overview of the project, setup instructions, and documentation.
```

---

## 🏁 Getting Started

1.  **Prerequisites**:
    *   Python 3.10+
    *   `pip` for package management
    *   `git` for cloning
    *   **Docker**: For sandboxed code evaluation. Ensure Docker Desktop (Windows/Mac) or Docker Engine (Linux) is installed and running. Visit [docker.com](https://www.docker.com/get-started) for installation instructions.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shyamsaktawat/OpenAlpha_Evolve.git
    cd OpenAlpha_Evolve
    ```

3.  **Set Up a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up Environment Variables (Crucial for API Keys)**:
    *   **This step is essential for the application to function correctly with your API keys.** The `.env` file stores your sensitive credentials and configuration, overriding the default placeholders in `config/settings.py`.
    *   Create your personal environment file by copying the example:
        ```bash
        cp .env_example .env
        ```


### LLM Configuration



8.  **Run OpenAlpha_Evolve!**
    Run the example task (Dijkstra's algorithm) with:
    ```bash
    python -m main examples/shortest_path.yaml
    ```
    Watch the logs in your terminal to see the evolutionary process unfold! Log files are also saved to `alpha_evolve.log` (by default).

8.  **Launch the Gradio Web Interface**
    Interact with the system via the web UI. To start the Gradio app:
    ```bash
    python app.py
    ```
    Gradio will display a local URL (e.g., http://127.0.0.1:7860) and a public share link if enabled. Open this in your browser to define custom tasks and run the evolution process interactively.

---

## 💡 Defining Your Own Algorithmic Quests!

Want to challenge OpenAlpha_Evolve with a new problem? It's easy! You can define your tasks in two ways:

### 1. Using YAML Files (Recommended)

Create a YAML file in the `examples` directory with the following structure:

```yaml
task_id: "aisp_llm_finetune_output_simple_001" # 对应输入的示例 ID
task_description: |
  简化版 AISP 输出示例：LLM 微调与评估结果。
  包含了微调后模型的引用、在验证集（或测试集）上的主要性能指标。

aisp_payload:
  task_id: "LLM_Finetune_Summarization_Qwen2_001" # 本次微调任务的唯一ID
  research_context:
    domain: "自然语言生成 - 文本摘要"
    objective: "微调 Qwen2-7B 模型，以提高其在新闻文章摘要任务上的 ROUGE-L 分数。"
    # background_knowledge: (可选, 此处留空)
    # constraints: (可选, 如最大训练时长、GPU型号限制等)
  input_data:
    base_model_id: "qwen/Qwen2-7B-Instruct" # 基础模型标识 (例如 Hugging Face Hub ID)
    dataset_references: # 数据集引用 (可以是路径、URL或内部ID)
      train_data: "s3://my-datasets/news_summarization/train.jsonl"
      validation_data: "s3://my-datasets/news_summarization/validation.jsonl"
      # test_data: (可选, 如果评估在另一独立步骤或使用验证集评估)
  parameters:
    fine_tuning_config:
      learning_rate: 0.00001 # 例如 1e-5
      batch_size_per_device: 4
      num_train_epochs: 1 # 简化场景，只训练1个epoch
      # optimizer: (可选, 默认为 AdamW)
      # lr_scheduler_type: (可选, 默认为 linear)
      # max_seq_length: (可选, 模型默认或根据数据调整) 512
    evaluation_config:
      metrics: ["rougeL", "bleu"] # 需要评估的指标
      # evaluation_strategy: (可选, "epoch" 或 "steps") "epoch"
      # eval_batch_size_per_device: (可选) 8
    # early_stopping_config: (可选, 此处留空表示不使用早停)

# AISP 通用输出格式的实际负载 (payload)
aisp_payload:
  task_id: "LLM_Finetune_Summarization_Qwen2_001" # 对应输入的任务ID
  execution_id: "exec_qwen2_sum_ft_20250527_001" # 本次执行的唯一ID
  result:
    optimized_model_reference: # 优化后模型的引用
      # model_id: (可选, 如果有内部模型注册库的ID) "Qwen2-7B-Summ-News-v1.0"
      checkpoint_url: "s3://my-finetuned-models/Qwen2_7B_Summarization_001/checkpoint-final/"
      # tokenizer_url: (可选, 如果分词器有变动或需单独提供) "s3://my-finetuned-models/Qwen2_7B_Summarization_001/tokenizer/"
    performance_metrics: # 在验证集/测试集上的表现
      rougeL: 0.452 # ROUGE-L F1 分数
      bleu: 0.381
      # validation_loss: (可选) 1.253
      # other_metrics: (可选) {...}
    summary_of_findings: |
      Qwen2-7B 模型经过1个epoch的微调后，在新闻摘要任务的验证集上取得了 ROUGE-L 0.452 的分数。
      初步结果显示模型对摘要任务有一定的适应性提升。
    # training_duration_hours: (可选) 2.5
    # full_logs_url: (可选) "s3://my-training-logs/Qwen2_7B_Summarization_001/all_logs.txt"
  confidence: 0.90 # (可选) 模块对本次结果有效性的置信度
  metadata:
    processing_time_seconds: 9000 # 总处理时长 (例如 2.5 小时)
    # resources_used: (可选, 简化版可省略详细资源，如GPU型号、数量等)
    # intermediate_steps_summary: (可选, 简化版可省略详细步骤)
    #   - "Data loading and preprocessing completed."
    #   - "Fine-tuning epoch 1/1 completed."
    #   - "Evaluation on validation set completed."
```

See the example in examples/shortest_path.yaml

### 2. Using Python Code (Legacy)

You can still define tasks programmatically using the `TaskDefinition` class:

```python
from core.task_definition import TaskDefinition

task = TaskDefinition(
    id="your_task_id",
    description="Your detailed problem description",
    function_name_to_evolve="your_function_name",
    input_output_examples=[
        {"input": [arg1, arg2], "output": expected_output},
        # More examples...
    ],
    allowed_imports=["module1", "module2"]
)
```

### Best Practices for Task Definition

Crafting effective task definitions is key to guiding OpenAlpha_Evolve successfully. Consider these tips:

*   **Be Clear and Unambiguous**: Write task descriptions as if you're explaining the problem to another developer. Avoid jargon where possible, or explain it clearly.
*   **Provide Diverse and Comprehensive Examples**: Your test cases are the primary way the agent verifies its generated code.
    *   Include typical use cases
    *   Cover edge cases (empty inputs, boundary values, etc.)
    *   Include examples that test different logical paths
    *   Use validation functions for complex checks
*   **Start Simple, Then Increase Complexity**: Break down complex problems into simpler versions first.
*   **Specify Constraints and Edge Cases**: Mention specific constraints and edge cases in the description.
*   **Define Expected Function Signature**: Clearly state the expected function name and parameters.
*   **Iterate and Refine**: Review and refine your task definition based on the agent's performance.

---

## 🔮 The Horizon: Future Evolution



---

## 🤝 Join the Evolution: Contributing

This is an open invitation to collaborate! Whether you're an AI researcher, a Python developer, or simply an enthusiast, your contributions are welcome.

*   **Report Bugs**: Find an issue? Please create an issue on GitHub!
*   **Suggest Features**: Have an idea to make OpenAlpha_Evolve better? Open an issue to discuss it!
*   **Submit Pull Requests**:
    *   Fork the repository.
    *   Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature-name`).
    *   Write clean, well-documented code.
    *   Add tests for your changes if applicable.
    *   Ensure your changes don't break existing functionality.
    *   Submit a pull request with a clear description of your changes!

Let's evolve this agent together!

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE.md` file for details.

---

## 🙏 Homage

OpenAlpha_Evolve is proudly inspired by the pioneering work of the Google DeepMind team on AlphaEvolve and other related research in LLM-driven code generation and automated discovery. This project aims to make the core concepts more accessible for broader experimentation and learning. We stand on the shoulders of giants.

---

*Disclaimer: This is an experimental project. Generated code may not always be optimal, correct, or secure. Always review and test code thoroughly, especially before using it in production environments.* 
