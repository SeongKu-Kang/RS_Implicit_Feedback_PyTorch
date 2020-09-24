# RS for Implicit Feedback in PyTorch
PyTorch implementations of collaborative filtering methods with implicit feedback


## 1. Implemented Methods
The implemented methods are as follows:
1. BPR, BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI'09
2. CML, Collaborative Metric Learning, WWW'17
3. MLP, Neural Collaborative Filtering, WWW'17
4. NeuMF, Neural Collaborative Filtering, WWW'17

## 2. Evaluation
### 2.1. Leave-One-Out (LOO) protocol
We provide the leave-one-out evaluation protocol widely used in recent research papers.
The protocol is as follows:
* For each test user
	1. randomly sample two positive (observed) items 
		- each of them is used for test/validation purpose.
	2. randomly sample 499 negative (unobserved) items
	3. evaluate how well each method can rank the test item higher than these sampled negative items.

### 2.2. Metrics
We provide three ranking metrics broadly adopted in many recent papers:  HR@N, NDCG@N, MRR@N.
The hit ratio simply measures whether the test item is present in the top-$N$ list, which is defined as follows:


![Large](https://latex.codecogs.com/svg.latex?\text{H}%20@%20N%20=%20\frac%20{%201%20}%20{%20|%20\mathcal%20{%20U%20}_{test}%20|%20}%20\sum%20_%20{%20u%20\in%20\mathcal%20{%20U}%20_{test}%20}%20\delta%20\left(%20p%20_%20{%20u%20}%20\leq%20\text%20{%20top%20}%20N%20\right) )

where $\delta ( \cdot )$ is the indicator function, $\mathcal { U }_{test}$ is the set of the test users, $p_u$ is the hit ranking position of the test item for the user $u$. 
On the other hand, the normalized discounted cumulative gain and the mean reciprocal rank are ranking position-aware metrics that put higher scores to the hits at upper ranks.
N@N and M@N are defined as follows:

![Large](https://latex.codecogs.com/svg.latex?\text{N}%20@%20N%20=%20\frac%20{%201%20}%20{%20|%20\mathcal%20{%20U%20}%20_{test}%20|%20}%20\sum%20_%20{%20u%20\in%20\mathcal%20{%20U%20}_{test}%20}%20\frac%20{%20\log%202%20}%20{%20\log%20\left(%20p%20_%20{%20u%20}%20+%201%20\right)%20}\text{,%20M}%20@%20N%20=%20\frac%20{%201%20}%20{%20|%20\mathcal%20{%20U%20}_{test}%20|%20}%20\sum%20_%20{%20u%20\in%20\mathcal%20{%20U%20}%20_{test}%20}%20\frac%20{%201%20}%20{%20p%20_%20{%20u%20}%20})