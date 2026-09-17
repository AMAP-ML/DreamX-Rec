# 📖 Overview

This repository collects a series of **generative recommendation models from DreamX-Rec**, spanning multi-task generative frameworks and efficient mixture-of-experts architectures:

*   **IntTravel**: Our foundational work that introduced a large-scale, real-world dataset and a generative framework for integrated multi-task travel recommendation.
*   **IntHQ**: The advanced successor to IntTravel, which identifies and resolves the "threefold collapse" in generative multi-task models with a novel architecture.
*   **IntBMoE**: An efficient mixture-of-experts architecture for generative recommendation, which uses block-conditioned expert composition to decouple expert participation, execution, and materialization.


# IntTravel: A Large-Scale Real-World Dataset  [![Data Set](https://img.shields.io/badge/Data-Set-green)](https://huggingface.co/datasets/GD-ML/IntTravel_dataset/tree/main)

We introduce **IntTravel**, the first large-scale public dataset for **integrated travel recommendation**, including **4.1 billion interactions from 163 million users with 7.3 million POIs**. Built upon this dataset, we introduce an end-to-end, **decoder-only generative framework for multi-task recommendation**. 

All data are collected from a leading provider of digital map, navigation and real-time traffic information in China. A small three-token sample is provided in `data_process/raw_data`, and a more comprehensive dataset is available from the linked dataset page. The code in `data_process` demonstrates how to construct model input sequences and labels from the original data.

### Information of POIs

The IntTravel dataset contains **7,291,872** POIs (Point of Interests) distributed across several major cities in China. Each POI is described by the following fields:

| Field | Description |
|:---|:---|
| POI ID | A unique identifier for each Point of Interest. |
| Normalized score | A 0-1 score reflecting the overall popularity of the POI. |
| Geographic ID | Identifier for the POI's geographic block. Same GIDs indicate geographical proximity. |
| Category ID | A numerical identifier for the Point of Interest's category. |
| Administrative Region ID | The identifier for the administrative region of the POI. |
| Coordinates | The spatial coordinates of the POI on a 2D plane. |

### User Profiles
The IntTravel dataset contains **162,815,861** users, each described by the following fields:

| Field | Description |
|:---|:---|
| User ID | A unique identifier assigned to each user. |
| Profile Feature 1 | The first profile feature. |
| ... | ... |
| Profile Feature 6 | The sixth profile feature. |


### User Interactions
The IntTravel dataset includes **4,129,827,011** user interaction events. Each event is characterized by the following fields:

| Field | Description |
|:---|:---|
| User ID | A unique identifier for the user who performed the interaction. |
| Timestamp | The time of the user interaction, recorded in milliseconds. |
| Action Type | A numerical ID representing the type of user behavior (e.g., click). |
| POI ID | The identifier of the Point of Interest involved in the interaction. |
| Geographic ID | The geographic block ID where the user was during the interaction. |
| Administrative Region ID | The administrative region ID where the user was during the interaction. |
| Weather | A numerical ID representing the weather condition during the interaction. |
| Travel Mode | A numerical ID for the user's chosen travel mode. |
| Via POI ID | The identifier for a way-point POI added by the user. |

# IntTravel: Generative Framework for Integrated Multi-Task Travel Recommendation  [![Paper Page](https://img.shields.io/badge/Paper-Page-blue)](https://arxiv.org/abs/2602.11664)

IntTravel incorporates information preservation, selection, and factorization to balance task collaboration with specialized differentiation, yielding substantial performance gains. IntTravel has been successfully deployed on Amap serving hundreds of millions of users.

![IntTravel multi-task framework](docs/assets/inttravel-framework.png)

IntTravel is **the first multi-task solution for generative recommendation**. We propose a bottom-up multi-task method to handle multiple tasks within a single generative model. The approach comprises three modules:

*   **Task-Guided Information Persistence (TIP)** ensures maximum propagation of task-relevant information in the decoder.
*   **Task-Specific Selective Gating (TSG)** enables each task to filter useful information from the decoder's output.
*   **Task-Aware Scenario Factorization (TSF)** empowers each task to factorize its output based on specific scenarios.




# IntHQ: Task-Interactive Hierarchical Query on Dual-Stream Representations for Generative Recommendation  [![Paper Page](https://img.shields.io/badge/Paper-Page-blue)](https://arxiv.org/abs/2608.09634)

Multi-task learning over heterogeneous data is fundamental to modern recommendation, while generative models are emerging as the backbone of next-generation recommenders. However, the integration of multi-task learning into the generative paradigm remains largely unexplored. Existing multi-task recommenders, in both discriminative and generative paradigms, extract task-relevant features from a single task-agnostic representation and wire tasks into a predefined conversion funnel. We show that this scheme is inherently prone to a threefold collapse. 
*   **Source collapse**, where task-specific signals are injected late and diluted in the shared latent space. 
*   **Relational collapse**, where task dependencies are either implicitly absorbed by the backbone or statically fixed by predefined funnels. 
*   **Hierarchical collapse**, where tasks depend on features at different scales and shift across training stages.

![IntHQ framework](docs/assets/inthq-framework.png)

**IntHQ** is a multi-task generative recommender with three components, each alleviating one collapse:

* **Dual-Stream Decoupling (DSD)** injects task identity into the computation stream early and separates the shared context stream from the task-specific stream, alleviating signal dilution.

* **Task-Interactive Modeling (TIM)** replaces the predefined funnel with explicit cross-task interaction, letting each task condition on the realized outcomes of its predecessors with learned, input-adaptive strength.

* **Hierarchical Querying (HQ)** lets each task gather multi-scale information across different layers at different training stages.

In offline evaluations, **IntHQ** consistently outperforms competitive encoder backbones under four representative task-head configurations. Deployed in production on Amap, serving hundreds of millions of users for travel recommendation, **IntHQ** yields a 1.60% relative UVCTR lift.


# IntBMoE: Efficient Mixture-of-Experts via Block-Conditioned Expert Composition  [![Paper Page](https://img.shields.io/badge/Paper-Page-blue)](https://arxiv.org/abs/XXXX.XXXXX)

IntBMoE is a block-conditioned mixture-of-experts architecture for dense expert participation with sparse block execution. The `intbmoe_demo` directory contains one shared BlockMoE implementation and task-specific examples for language modeling, image classification, and POI recommendation.

Mixture-of-Experts (MoE) scales model capacity, but existing designs cannot set three quantities independently: **participation**, the number of experts that contribute knowledge to a token; **execution**, the number of experts actually computed; and **materialization**, the number of expert-sized parameter sets that must be built and stored. This coupling creates three common trade-offs:

- **Sparse execution limits participation.** Sparse-routing methods compute only a few selected experts, so every other expert is excluded from the token's output and learning signal.
- **Full participation requires dense execution.** Dense output-mixing methods use the whole expert pool, but must evaluate every expert for every token.
- **Single-expert execution increases materialization.** Parameter-merging methods execute one composed expert, but every distinct routing decision can require another expert-sized set of composed weights.

**IntBMoE** separates how expert transformations are constructed from how they are executed on tokens. It addresses these trade-offs with two core designs:

- **Block-conditioned expert composition** uses a small learned codebook and a shared hypernetwork to merge every layer's full pool of expert bases into a bounded set of reusable blocks. Every composed block therefore receives pool-wide expert participation, while the fixed codebook bounds parameter materialization.
- **Sparse block execution with Dual-Path Residual Gating (DPRG)** routes each token to only its Top-k blocks. Within each block, independently composed value and gate paths interact multiplicatively, increasing expressiveness without enlarging the expert pool.

Across MiniPile, ImageNet-1K, and IntTravel, IntBMoE consistently outperforms the evaluated sparse and dense MoE baselines. It is also deployed in AMap's generative recommendation system, where the paper reports a 2.4% relative UVCTR improvement under a 60 ms serving budget.

![IntBMoE framework](docs/assets/intbmoe-framework.png)


## Repository layout

- `data_process/`: the official three-token preprocessing example and sample data.
- `inthq_demo/`: the IntHQ dual-stream encoder with DSFNet.
- `inttravel_demo/`: the IntTravel HyperConnection encoder with DSFNet.
- `intbmoe_demo/`: the shared BlockMoE implementation with examples for language modeling, image classification, and POI recommendation.

Both demos consume S/I/F sessions: scenario features are merged into S, POI intention features stay on I, and feedback features stay on F.

Each demo directory also contains its own README and dependency specification.

# 📚 Citation

If you find our papers and code helpful for your research, please consider starring our repository ⭐ and citing our work ✏️.

```bibtex
@article{yan2026inttravel,
  title={IntTravel: A Real-World Dataset and Generative Framework for Integrated Multi-Task Travel Recommendation},
  author={Yan, Huimin and Xu, Longfei and Sun, Junjie and Liu, Zheng and Luo, Wei and Liu, Kaikui and Chu, Xiangxiang},
  journal={arXiv preprint arXiv:2602.11664},
  year={2026}
}
@article{sun2026inthq,
  title={IntHQ: Task-Interactive Hierarchical Query on Dual-Stream Representations for Generative Recommendation},
  author={Sun, Junjie and Xu, Longfei and Yan, Huimin and Luo, Wei and Liu, Kaikui and Chu, Xiangxiang},
  journal={arXiv preprint arXiv:2608.09634},
  year={2026}
}
@article{cheng2026intbmoe,
  title={IntBMoE: Efficient Mixture-of-Experts via Block-Conditioned Expert Composition},
  author={Cheng, Ran and Xu, Longfei and Liu, Zheng and Liu, Kaikui and Chu, XiangXiang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

# License

Released under the [Apache License 2.0](LICENSE).



