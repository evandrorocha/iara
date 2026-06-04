import os

base_dir = "results/trainings/tests/svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
for fold in range(10):
    model_path = os.path.join(base_dir, "model", f"fold_{fold}", "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5_multiclass.pkl")
    print(f"Fold {fold} model: {model_path} -> {'EXISTS' if os.path.exists(model_path) else 'MISSING'}")
