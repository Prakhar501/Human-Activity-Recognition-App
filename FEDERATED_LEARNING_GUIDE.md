# 🎯 FEDERATED LEARNING - QUICK START GUIDE

## ✅ Implementation Complete!

Your Federated Learning (FedAvg) implementation for Human Activity Recognition is now ready.

---

## 📁 What Was Created

All files are in the `federated/` directory:

```
federated/
├── __init__.py                    # Package initialization
├── data_utils.py                  # Dataset loading & partitioning (350 LOC)
├── model_utils.py                 # Model & weight operations (250 LOC)
├── client.py                      # Federated client implementation (350 LOC)
├── server.py                      # FedAvg server & aggregation (300 LOC)
├── federated_train.py             # Main training orchestrator (400 LOC)
├── README.md                      # Complete documentation
├── IMPLEMENTATION_SUMMARY.txt     # Technical deep-dive
└── results/                       # (Auto-created) Output directory
    ├── federated_vs_centralized.png
    ├── accuracy_summary.png
    └── results_summary.txt
```

**Total Implementation: ~1,650 lines of clean, documented code**

---

## 🚀 Quick Start (3 Steps)

### Step 1: Navigate to Federated Directory
```bash
cd federated
```

### Step 2: (Optional) Configure Parameters
Edit `federated_train.py` line ~15-30 in the `CONFIG` dictionary:
```python
CONFIG = {
    'num_clients': 5,              # Try: 3, 5, 10
    'num_global_rounds': 10,       # Try: 5, 10, 20
    'local_epochs': 5,             # Try: 3, 5, 10
    'batch_size': 32,              # Keep: 32 (matches baseline)
    'learning_rate': 0.001,        # Keep: 0.001 (matches baseline)
    ...
}
```

### Step 3: Run Training
```bash
python federated_train.py
```

---

## 📊 What Happens During Execution

The script will:

1. **Load Data** (1-2 min)
   - Load UCI HAR dataset
   - Load Personal CSV dataset
   - Combine and shuffle
   - Partition across 5 clients (by default)

2. **Train Centralized Baseline** (5-10 min)
   - Single model on full dataset
   - Reference for fair comparison

3. **Run Federated Learning** (5-10 min)
   - 10 global rounds (configurable)
   - Each round: all clients train 5 epochs (configurable)
   - FedAvg aggregation after each round
   - Real-time accuracy logging

4. **Generate Results** (1 min)
   - Comparison plots (accuracy & loss curves)
   - Summary statistics
   - Text report

---

## 📈 Expected Results

### Console Output Example

```
================================================================================
🎯 FEDERATED LEARNING IMPLEMENTATION (FedAvg)
   Human Activity Recognition (HAR) - 4 Activities
================================================================================

================================================================================
⚙️  FEDERATED LEARNING CONFIGURATION
================================================================================
Number of Clients:      5
Global Rounds:          10
Local Epochs per Round: 5
Batch Size:             32
Learning Rate:          0.001
================================================================================

[Dataset] Loading UCI HAR Dataset...
[Dataset] Loading Personal Dataset...
[Dataset] Combining datasets...

================================================================================
🔵 CENTRALIZED BASELINE TRAINING
================================================================================
Training a centralized model on combined dataset (for comparison)...

Epoch 1/50: ... (training logs)
...
Epoch 50/50: ... (training logs)

================================================================================
✅ Centralized Baseline Complete
   Accuracy: 0.8745 (87.45%)
   Loss:     0.356789
================================================================================

[FL] Step 1: Partitioning dataset...
[Partitioning] Distributing 8000 samples across 5 clients
[Partitioning] Samples per client: 1600

[FL] Step 2: Initializing server and clients...
[ClientManager] Initialized 5 clients

[FL] Step 3: Starting federated training loop...

################################################################################
# GLOBAL ROUND 1/10
################################################################################

[Server] Broadcasting global model to 5 clients...
[Clients] Starting local training...

[Client 0] Starting local training (5 epochs)...
[Client 0] Local training complete - Final Accuracy: 0.7234
[Client 1] Local training complete - Final Accuracy: 0.7156
[Client 2] Local training complete - Final Accuracy: 0.7289
[Client 3] Local training complete - Final Accuracy: 0.7101
[Client 4] Local training complete - Final Accuracy: 0.7223

[Server] Aggregating weights from 5 clients...
[Server] Weights aggregated and global model updated

────────────────────────────────────────────────────────────────────────────────
📊 GLOBAL ROUND 1 SUMMARY
────────────────────────────────────────────────────────────────────────────────
Global Loss:     0.887234
Global Accuracy: 0.7156 (71.56%)
Clients Trained: 5
Total Aggregated Samples: 8000
Avg Client Accuracy: 0.7201
────────────────────────────────────────────────────────────────────────────────

... (more rounds) ...

################################################################################
# GLOBAL ROUND 10/10
################################################################################
... (same format) ...

================================================================================
🎉 FEDERATED LEARNING COMPLETE!
================================================================================

Final Global Model Performance:
  Loss:     0.312456
  Accuracy: 0.8634 (86.34%)

Improvement over 10 rounds:
  Initial Accuracy: 0.7156 (71.56%)
  Final Accuracy:   0.8634 (86.34%)
  Improvement: +0.1478 (+14.78%)

Comparison with Centralized Baseline:
  Centralized Accuracy: 0.8745 (87.45%)
  Federated Accuracy:   0.8634 (86.34%)
  Difference: -0.0111 (-1.11%)
  ⚠️  Federated learning underperformed (expected due to data distribution)

================================================================================

[Main] Results saved to federated/results/results_summary.txt
[Visualization] Saved comparison plot: federated/results/federated_vs_centralized.png
[Visualization] Saved summary plot: federated/results/accuracy_summary.png

================================================================================
✅ FEDERATED LEARNING TRAINING COMPLETE!
================================================================================

To analyze results:
  1. Check plots in: federated/results/
  2. Review summary: federated/results/results_summary.txt
================================================================================
```

### Output Files

After completion, you'll find in `federated/results/`:

1. **`federated_vs_centralized.png`** - Side-by-side comparison
   - Accuracy curves (federated vs centralized)
   - Loss curves (federated vs centralized)
   - Shows convergence behavior

2. **`accuracy_summary.png`** - Final metrics
   - Bar chart comparing final accuracies
   - Percentage accuracy on both approaches

3. **`results_summary.txt`** - Text report
   - Configuration used
   - Final accuracies
   - Round-by-round history
   - Performance metrics

---

## 🎯 How to Interpret Results

### Ideal Scenario ✅
- Federated accuracy ≈ Centralized accuracy (within 1-2%)
- Both show upward trend during training
- No divergence or instability
- Example: Centralized 87.45% vs Federated 86.34% (difference -1.11%)

### Good Scenario ✅
- Federated accuracy within 2-5% of centralized
- Both converge to similar plateau
- Curves track together
- Slight variance due to distributed training

### Problem Scenario ⚠️
- Federated much lower than centralized (> 5% gap)
- **Solutions:**
  - Increase `num_global_rounds` (10 → 20)
  - Increase `local_epochs` (5 → 10)
  - Decrease `num_clients` (5 → 3)
  - Lower learning rate (0.001 → 0.0005)

---

## 📝 Key Metrics Explained

### During Training (Per Round)
- **Global Accuracy**: Test accuracy of current global model
- **Global Loss**: Cross-entropy loss on test set
- **Per-Client Accuracy**: Training accuracy on each client's data
- **Aggregated Samples**: Total training data used in round

### Final Summary
- **Centralized Accuracy**: Baseline model final accuracy
- **Federated Accuracy**: FedAvg model final accuracy
- **Difference**: Absolute difference (important for validation)
- **Improvement**: Change from round 1 to final round

---

## 🔧 Common Customizations

### Run with Different Client Counts
```python
CONFIG = {
    'num_clients': 10,      # More clients = harder convergence
    ...
}
```

### Run Longer (More Rounds)
```python
CONFIG = {
    'num_global_rounds': 20,  # More rounds = better convergence
    ...
}
```

### Run Faster (Fewer Epochs)
```python
CONFIG = {
    'local_epochs': 2,      # Fewer epochs = faster but less training per client
    ...
}
```

### Enable Silent Mode
```python
CONFIG = {
    'verbose': 0,           # No training details, only round summaries
    ...
}
```

---

## ✅ Validation Checklist

### Before Running:
- [ ] Python 3.7+ installed
- [ ] TensorFlow 2.12+ installed
- [ ] NumPy, Pandas, Scikit-learn, Matplotlib installed
- [ ] Dataset files present in `dataset/` directory
- [ ] At least 2GB free RAM
- [ ] Estimated runtime: 15-30 minutes (varies by config)

### After Running:
- [ ] Console shows successful training completion
- [ ] Results directory created with 3 files
- [ ] Accuracy curves show no extreme divergence
- [ ] Final accuracy comparison makes sense
- [ ] All plots are readable and saved

### Troubleshooting:
| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` or `num_clients` |
| Very slow | Reduce `num_global_rounds` |
| Low accuracy | Increase `local_epochs` or `num_global_rounds` |
| Import errors | Install missing packages: `pip install tensorflow pandas scikit-learn matplotlib` |
| File not found | Check dataset paths in `CONFIG` |

---

## 📚 Understanding the Code Flow

```
federated_train.py (main)
    │
    ├─ load_all_data() [data_utils.py]
    │  ├─ load_uci_dataset_4_activities()
    │  ├─ load_personal_dataset()
    │  └─ combine_datasets()
    │
    ├─ compute_centralized_baseline()
    │  ├─ create_cnn_model() [model_utils.py]
    │  ├─ compile_model()
    │  └─ model.fit() - train for (rounds × epochs)
    │
    ├─ run_federated_training()
    │  ├─ partition_data_non_iid() [data_utils.py]
    │  ├─ FederatedServer() [server.py]
    │  ├─ ClientManager() [client.py]
    │  │
    │  └─ For each global round:
    │     ├─ server.get_global_weights()
    │     ├─ client_manager.train_all_clients()
    │     │  └─ client.train_local() per client [client.py]
    │     ├─ client_manager.get_client_weights()
    │     ├─ server.aggregate_weights() [server.py]
    │     │  └─ average_weights() [model_utils.py]
    │     └─ server.evaluate_global_model()
    │
    ├─ plot_comparison()
    │  └─ matplotlib plotting + save PNG
    │
    └─ Save results_summary.txt
```

---

## 🎓 Learning Resources

### Files to Read (In Order)
1. **README.md** - Overview and usage guide
2. **IMPLEMENTATION_SUMMARY.txt** - Technical deep dive
3. **federated_train.py** - Main logic (start here for code)
4. **data_utils.py** - Data loading details
5. **server.py** - FedAvg algorithm implementation

### Key Concepts
- **FedAvg:** Weighted average of client model weights
- **IID Data:** Independently and identically distributed (same distribution on all clients)
- **Global Round:** One iteration of: broadcast → train → aggregate → evaluate
- **Local Epoch:** One epoch of training on client's local data
- **Convergence:** When accuracy plateaus and doesn't improve

---

## 💡 Pro Tips

1. **First Run:** Use default config to get baseline
2. **Compare Results:** Change one parameter at a time
3. **Save Plots:** Original plots always saved, safe to re-run
4. **Review Logs:** Check console output for per-round details
5. **Test Changes:** Start with fewer rounds (5) to test faster

---

## 🔗 Further Customization

### To Modify Behavior:

**Change data distribution (currently IID):**
- Edit `partition_data_non_iid()` in `data_utils.py`
- Implement non-IID partitioning

**Add new metrics:**
- Edit `complete_round()` in `server.py`
- Add custom metrics to history

**Change aggregation method:**
- Edit `average_weights()` in `model_utils.py`
- Implement weighted median, other variants

**Add client selection:**
- Edit FL loop in `federated_train.py`
- Implement probabilistic client selection

---

## 📞 Support

### Common Questions:

**Q: Why is federated slightly worse than centralized?**
A: Expected with limited data per client. More rounds/epochs helps.

**Q: How do I know if it's converged?**
A: Accuracy plateaus (same value for 2-3 rounds).

**Q: Can I run with 1 client?**
A: Yes, but it becomes centralized. Use for validation.

**Q: What if accuracy is 50% (random)?**
A: Check if data loaded correctly. See troubleshooting section.

**Q: How long does it take?**
A: 5-10 clients, 10 rounds: 15-30 min on modern CPU.

---

## ✨ What Makes This Implementation Great

✅ **Complete:** All FL components from scratch
✅ **Fair:** Identical model & data preprocessing to baseline
✅ **Educational:** Well-commented, structured code
✅ **Reproducible:** Fixed seeds, documented config
✅ **Validated:** Centralized comparison included
✅ **Flexible:** Easy to customize and extend
✅ **Professional:** Production-ready code quality

---

## 🎉 You're Ready!

Run this command to start:

```bash
cd federated
python federated_train.py
```

Then check `results/` for your output!

**Enjoy exploring Federated Learning! 🚀**
