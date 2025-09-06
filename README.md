# csci653-as01
## A. Problem Description
We focus on the **Amazon Reviews dataset**, which contains millions of user–product interactions with ratings, reviews, and metadata.  
Our goal is to analyze which product features (e.g., price, review count, star rating, sentiment) are most correlated with purchases.  
Due to the dataset’s large scale, **high-performance computing (HPC)** is needed for efficient preprocessing, model training, and correlation analysis.

---

## B. Methods & HPCS
- **Algorithms**: Alternating Least Squares (ALS), Neural Collaborative Filtering (NCF), and sequence-based models (GRU4Rec, SASRec).  
- **Evaluation**: HR@K, NDCG@K, Recall@K.  
- **HPC Techniques**: PyTorch Distributed Data Parallel (DDP), Horovod, MPI + Slurm for distributed runs, OpenMP for CPU preprocessing, and mixed-precision training.  
- **Visualization**: accuracy curves, scaling plots, and feature correlation heatmaps.

---

## C. Expected Results
- **Accuracy**: Competitive HR@10 and NDCG@10 compared to published baselines.  
- **Performance**: Clear speedup from single-GPU → multi-GPU → multi-node; efficiency measured via strong/weak scaling.  
- **Artifacts**: Training scripts, Slurm job files, logs, and plots (accuracy, scaling efficiency, feature analysis).

