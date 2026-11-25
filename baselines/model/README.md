# Model of the previous study

The detector M_HAR is the activity classifier of the previous study (Rios et al., arXiv:2609.19359), used unchanged.
Its two files are not part of this repository. To run `baselines/b1_hgb_state.py` and `analysis/compute_cost.py`, place them here:

```
baselines/model/hgb_both_feet.joblib
baselines/model/hgb_both_feet_meta.joblib
```

They were saved with scikit-learn 1.6.1, which is pinned in `requirements.txt`.
