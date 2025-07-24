# CBTM-NCD: Clustering-Based Tail Class Mitigation for New-Class Discovery

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official code for **"Clustering-Based Tail Class Mitigation for New-Class Discovery"**, accepted at **ACM MM 2025**.

**Authors**:  
Zelei Wu, Xulun Ye*, Jieyu Zhao  
Ningbo University, China  
📧 Contact: [2311100309@nbu.edu.cn](mailto:2311100309@nbu.edu.cn)  
🔗 Paper: [ACM Digital Library](https://doi.org/10.1145/3746027.3755193)  
🔗 Code: [GitHub Repo](https://github.com/wuzelei123/CBTM-NCD)

---

## 🧠 Abstract

Open-world semi-supervised learning (OWSSL) enables models to discover novel classes from unlabeled data, enhancing their generalization ability. However, most OWSSL approaches assume balanced data, which is unrealistic. In real-world long-tailed scenarios, novel category discovery becomes more challenging due to:

- Insufficient feature representation of tail classes  
- Difficulty in discovering unknown categories  
- Severe class imbalance

We propose **CBTM-NCD**, a **Class-Balanced Representation and Recognition Framework** that includes:

- 📊 A **Variational Dirichlet Process (VDP)** for tail class clustering  
- 🌌 A **diffusion-based generative module** for class-balanced pseudo-sample generation  
- ♻️ A **two-stage optimization strategy** to balance old/new class learning and enhance separability

---

## 🏗️ Framework Overview

![CBTM-NCD Framework](https://user-images.githubusercontent.com/your-placeholder-path/framework.png)

CBTM-NCD contains:

1. **Tail Class Clustering** via VDP  
2. **Sample Generation** using Diffusion Model  
3. **Novel Class Discovery** via Contrastive Clustering

---

## 📦 Installation

```bash
git clone https://github.com/wuzelei123/CBTM-NCD.git
cd CBTM-NCD
pip install -r requirements.txt
