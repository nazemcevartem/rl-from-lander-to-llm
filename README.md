<div align="center">

# RL Methods: Stable-Baselines3 & TRL

### От классических алгоритмов на Lunar Lander до выравнивания LLM с помощью DPO и GRPO

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-2.8-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![TRL](https://img.shields.io/badge/TRL-0.14-orange.svg)](https://github.com/huggingface/trl)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2-9cf.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## О проекте

Практическое исследование методов обучения с подкреплением — от классических алгоритмов (PPO, DQN, DDPG, SAC) на задаче управления посадкой лунохода **Lunar Lander** до современных подходов выравнивания языковых моделей (DPO, GRPO) с использованием библиотеки **TRL** от Hugging Face.

Проект охватывает два ключевых направления современного RL:

- **Классическое RL** — обучение агентов в средах Gymnasium через Stable-Baselines3 и кастомную реализацию GRPO на PyTorch
- **RLHF / LLM Alignment** — прямая оптимизация предпочтений (DPO) и групповая относительная оптимизация политики (GRPO) для LLM

<details>
<summary><b>Зачем это нужно?</b></summary>

Классические алгоритмы RL (PPO, DQN и др.) долгое время были основой для обучения агентов в играх, робототехнике и системах управления. С появлением ChatGPT выяснилось, что те же принципы — в частности PPO — могут быть использованы для «выравнивания» языковых моделей с человеческими предпочтениями (RLHF). Новые методы, такие как DPO и GRPO, позволяют обходиться без явной модели награды, обучая LLM напрямую на парах «хороший/плохой ответ» или на группах сгенерированных ответов. Этот проект демонстрирует всю цепочку: от Lunar Lander до тонкой настройки Qwen2.5.

</details>

---

## Архитектура проекта

```
RL_Methods_stable_baselines3_and_trl.ipynb   # Основной ноутбук (Google Colab)
```

---

## Методы

### Часть I. Классическое RL — Lunar Lander

Все агенты обучаются управлять посадочным модулем в среде [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) из Gymnasium. Среда представлена в двух вариантах: с дискретным (4 действия) и непрерывным (2 непрерывных действия) пространством действий.

| Алгоритм | Среда | Тип действий | Библиотека | Timesteps |
|:--------:|-------|:------------:|:----------:|:---------:|
| **PPO** | LunarLander-v3 | Дискретные | Stable-Baselines3 | 300K |
| **DQN** | LunarLander-v3 | Дискретные | Stable-Baselines3 | 300K |
| **QRDQN** | LunarLander-v3 | Дискретные | sb3_contrib | 300K |
| **DDPG** | LunarLanderContinuous-v3 | Непрерывные | Stable-Baselines3 | 500K |
| **SAC** | LunarLanderContinuous-v3 | Непрерывные | Stable-Baselines3 | 300K |
| **GRPO** | LunarLander-v3 | Дискретные | PyTorch (с нуля) | 100K |

<details>
<summary><b>Краткое описание каждого алгоритма</b></summary>

**PPO (Proximal Policy Optimization)** — actor-critic алгоритм, который оптимизирует политику с ограничением на размер обновления (clipped surrogate objective). Считается стандартом «золотой середины» по соотношению стабильности, скорости сходимости и простоты настройки гиперпараметров.

**DQN (Deep Q-Network)** — классический value-based алгоритм. Использует нейросеть для аппроксимации Q-функции, replay buffer для разрыва корреляций и target network для стабильности обучения.

**QRDQN (Quantile Regression DQN)** — расширение DQN, которое моделирует полное распределение наград через квантили вместо ожидаемого значения. Позволяет лучше справляться со стохастическими средами за счёт multi-step learning (n_steps=5).

**DDPG (Deep Deterministic Policy Gradient)** — off-policy actor-critic для непрерывных пространств действий. Использует детерминированную политику и шум для исследования среды (NormalActionNoise).

**SAC (Soft Actor-Critic)** — state-of-the-art алгоритм для непрерывных сред. Добавляет энтропийный бонус в целевую функцию, что обеспечивает автоматический баланс между exploration и exploitation (ent_coef='auto').

**GRPO (Group Relative Policy Optimization)** — кастомная реализация на чистом PyTorch. Идея: запуск группы параллельных сред (G=8), вычисление advantage относительно среднего по группе и обновление политики без value-функции. Тот же принцип, который используется в DeepSeek-R1 для обучения LLM.

</details>

---

### Часть II. LLM Alignment — TRL

| Метод | Модель | Датасет | Задача |
|:-----:|--------|---------|--------|
| **DPO** | Qwen2.5-1.5B-Instruct (LoRA, 4-bit) | UltraFeedback (2K примеров) | Предпочтения: chosen vs rejected |
| **GRPO** | Qwen2.5-1.5B-Instruct (LoRA, 4-bit) | GSM8K (7.4K примеров) | Математическое рассуждение |

<details>
<summary><b>Подробности настройки</b></summary>

**DPO (Direct Preference Optimization):**
- Модель: `unsloth/Qwen2.5-1.5B-Instruct` с LoRA (r=16, alpha=32)
- Квантизация: 4-bit (load_in_4bit=True)
- Датасет: HuggingFaceH4/ultrafeedback_binarized, 2000 примеров
- Параметры: beta=0.1, lr=5e-6, batch_size=4, gradient_accumulation=4, max_steps=150
- Формат обучения: пары (prompt, chosen_response, rejected_response)

**GRPO via TRL:**
- Модель: `unsloth/Qwen2.5-1.5B-Instruct` с LoRA (r=32, alpha=64)
- Датасет: openai/gsm8k (школьные математические задачи)
- Reward functions: format_reward (regex-проверка тегов) + correctness_reward (проверка ответа)
- Параметры: num_generations=4, lr=5e-6, batch_size=1, gradient_accumulation=4
- Особенности: генерация G=4 ответов на каждый промпт, сравнение по reward, обновление политики по групповому advantage

</details>

---

## Быстрый старт

### В Google Colab (рекомендуется)

Нажмите на кнопку и следуйте инструкциям в ноутбуке:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Локальный запуск

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/<your-username>/RL-Methods-SB3-TRL.git
cd RL-Methods-SB3-TRL

# 2. Создайте виртуальное окружение
conda create -n rl_methods python=3.10 -y
conda activate rl_methods

# 3. Установите зависимости
pip install gymnasium[box2d,lunar-lander] stable-baselines3[extra] sb3_contrib
pip install shimmy pyvirtualdisplay unsloth "trl<0.15.0" peft accelerate bitsandbytes

# 4. Откройте ноутбук
jupyter notebook RL_Methods_stable_baselines3_and_trl.ipynb
```

> **GPU**: Для части с LLM (DPO, GRPO через TRL) потребуется GPU с минимум 16 GB VRAM (tested on Tesla T4).

---

## Стек технологий

| Категория | Технологии |
|-----------|-----------|
| **RL фреймворки** | Stable-Baselines3, sb3_contrib, Gymnasium |
| **LLM фреймворки** | TRL, Unsloth, PEFT (LoRA), Transformers |
| **Deep Learning** | PyTorch, bitsandbytes (4-bit квантизация) |
| **Среды** | LunarLander-v3, LunarLanderContinuous-v3, GSM8K, UltraFeedback |
| **Инфраструктура** | Google Colab, TensorBoard, Weights & Biases |

---

## Требования

- Python 3.10+
- PyTorch 2.x с CUDA
- GPU (T4 / V100 / A100) для LLM-части
- ~16 GB VRAM для обучения Qwen2.5-1.5B в 4-bit

---

## Структура ноутбука

```
1. Установка зависимостей и настройка окружения
   └── gymnasium, stable-baselines3, box2d, pyvirtualdisplay

2. Классическое RL (Stable-Baselines3)
   ├── PPO  — LunarLander-v3 (дискретные действия)
   ├── DQN  — LunarLander-v3 (дискретные действия)
   ├── QRDQN — LunarLander-v3 (дискретные действия, sb3_contrib)
   ├── DDPG — LunarLanderContinuous-v3 (непрерывные действия)
   └── SAC  — LunarLanderContinuous-v3 (непрерывные действия)

3. GRPO с нуля (PyTorch)
   ├── Параллельные среды (G=8)
   ├── PolicyNetwork (MLP: 8→128→128→4)
   └── Групповой advantage без value-функции

4. LLM Alignment (TRL + Unsloth)
   ├── DPO — UltraFeedback + Qwen2.5-1.5B
   └── GRPO — GSM8K + Qwen2.5-1.5B (format + correctness rewards)
```

---

## Полезные ресурсы

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [DPO Paper (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [GRPO Paper (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)

---

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).
