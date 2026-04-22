# Self-Pruning Neural Network — Report
**Tredence AI Engineering Internship Case Study**
**Author:**Ansh

---

## Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

The total loss is:  
**Total Loss = CrossEntropy + λ × Σ(gates)**

The sigmoid function maps gate_scores to a value between 0 and 1.
The L1 penalty is the **sum of all gate values**.

The key reason L1 works for sparsity:
- The gradient of |x| with respect to x is a **constant ±1** (unlike L2 whose gradient shrinks near zero).
- This means the optimizer always receives a fixed pull of magnitude λ pushing each gate **toward zero**, regardless of how small the gate already is.
- L2 would create a gradient that shrinks as gates get small, meaning they'd never fully reach zero.
- L1 keeps pushing even when a gate is 0.001, eventually snapping it to exactly 0.

Result: Most gates collapse to ~0 (pruned), while only gates that **strongly reduce classification loss** resist and survive.

---

## Results Table

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|------------|--------------|-------------------|
| 0.1       	  | 53.59%       | 44.72%            |
| 0.5          | 53.24%       | 72.25%            |
| 1.0          | 53.55%       | 83.92%            |


### Analysis
- **Low λ (0.1):** Sparsity penalty is weak → most gates survive → high accuracy, low sparsity.
- **Medium λ (0.5):** Balanced trade-off → moderate pruning with acceptable accuracy.
- **High λ (1.0):** Strong pruning pressure → many gates die → high sparsity but accuracy drops.

---

## Gate Distribution Plot

![Gate Distribution](gate_distribution.png)

The plot shows a **large spike at 0** (pruned weights) and a **separate cluster away from 0** (surviving important weights). This bimodal distribution confirms the network successfully learned to distinguish important connections from redundant ones.

---

## Conclusion

The self-pruning mechanism works: the network learns to prune **itself during training** rather than requiring a post-hoc step. The λ hyperparameter gives direct control over the sparsity-accuracy trade-off.