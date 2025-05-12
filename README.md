TAG: Dialogue Summarization Based on Topic Segmentation and Graph Structures
# Dialogue Summarization with Static-Dynamic Structure Fusion Graph

This repository contains the official source code for the paper:  
**"Dialogue Summarization with Static-Dynamic Structure Fusion Graph" (SDDS)**.

## 📌 Introduction

Dialogue summarization is one of the most challenging and intriguing tasks in text summarization, and it has drawn increasing attention in recent years.

Due to the dynamic and interactive nature of dialogues — and the dispersed flow of information across multiple utterances from different speakers — many existing methods utilize static graph structures precomputed by external linguistic toolkits to model conversations.

However, such approaches heavily rely on the accuracy of these tools, and the static graph construction is disconnected from the representation learning phase. This limits the model's ability to adapt to the downstream summarization task.

In this work, we propose **SDDS**, a **Static-Dynamic Structure Fusion Graph** model for dialogue summarization. It integrates prior knowledge from linguistic tools and implicit knowledge from pretrained language models (PLMs). SDDS adaptively adjusts graph weights and learns graph structures end-to-end under summarization supervision.

---

## ⚙️ Setup

Our code is primarily based on 🤗 [Transformers](https://github.com/huggingface/transformers).

### 🧩 Install dependencies

Make sure to install PyTorch corresponding to your CUDA version first, then:

```bash
pip install transformers==4.8.2 \
            py-rouge \
            nltk \
            numpy \
            datasets \
            stanza \
            dgl
