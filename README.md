
# TAG: Dialogue Summarization Based on Topic Segmentation and Graph Structures


## 📌 Introduction
In recent years, dialogue summarization has emerged as a rapidly growing area of research in natural language processing. Dialogue summarization is challenging due to dispersed key information, redundant expressions, ambiguous topic identification, and difficult content selection. 
To address these challenges, we propose an innovative approach to dialogue summarization that integrates topic segmentation and graph-structured modeling. Specifically, we first perform topic segmentation of the dialogue through clustering and quantify the key information in each utterance, thereby capturing the dialogue topics more effectively. Then, a redundancy graph and a keyword graph are constructed to suppress redundant information and extract key content, thereby enhancing the conciseness and coherence of the summary. Evaluations were conducted on the DialogSum, SAMSum, CSDS, and NaturalConv datasets. The experimental results demonstrate that the proposed method significantly outperforms existing benchmark models in terms of summary accuracy and information coverage. The Rouge-1 scores achieved were 48.03%, 53.75%, 60.78%, and 81.48%, respectively, validating its effectiveness in the dialogue summarization task.

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
