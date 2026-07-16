"""
System Inference Speed Benchmark
================================
Measures the latency of:
1. Image Model (CNN+QNN)
2. Text Model (Hybrid BioBERT)
3. Ensemble Fusion (MLE)
4. Total End-to-End Local Execution

Author: NLP Project Team
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from backend.models.text import TextModel
from backend.models.image import ImageModel
from backend.services.inference import EnsembleInference

def main():
    # 1. Config
    DATASET_ROOT = 'dataset/Image Dataset'
    TEXT_MODEL_PATH = 'models/text/overhaul/best_biobert_overhaul.pth'
    TEXT_METADATA_PATH = 'models/text/overhaul/overhaul_metadata.pkl'
    IMAGE_MODEL_PATH = 'models/image/cnn_qnn_best.pt'
    IMAGE_MAP_PATH = 'models/image/label_mapping.pkl'
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_SAMPLES = 50
    WARMUP_RUNS = 5

    # 2. Load Models
    print(f"🚀 Benchmarking on {DEVICE}...")
    print("Loading models...")
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=TEXT_METADATA_PATH, device=DEVICE)
    image_model = ImageModel(model_path=IMAGE_MODEL_PATH, label_mapping_path=IMAGE_MAP_PATH, device=DEVICE)
    ensemble = EnsembleInference(alpha=0.65)
    
    # 3. Load Sample Data
    df = pd.read_csv('data/cleaned_multimodal_plant_data.csv').sample(NUM_SAMPLES, random_state=42)
    
    # 4. Benchmarking
    image_latencies = []
    text_latencies = []
    fusion_latencies = []
    total_latencies = []
    
    print(f"Warming up ({WARMUP_RUNS} runs)...")
    sample_row = df.iloc[0]
    sample_img = os.path.join(DATASET_ROOT, 'PlantVillage', sample_row['LABEL'], sample_row['FILENAME'])
    sample_text = sample_row['cleaned_text']
    
    for _ in range(WARMUP_RUNS):
        image_model.predict(sample_img)
        text_model.predict(sample_text)
        
    print(f"Benchmarking {NUM_SAMPLES} samples...")
    for _, row in tqdm(df.iterrows(), total=NUM_SAMPLES):
        img_path = os.path.normpath(os.path.join(DATASET_ROOT, 'PlantVillage', row['LABEL'], row['FILENAME']))
        text = row['cleaned_text']
        
        # End-to-end measure
        start_total = time.perf_counter()
        
        # Image
        start_img = time.perf_counter()
        img_probs = image_model.predict(img_path)[0]
        end_img = time.perf_counter()
        
        # Text
        start_txt = time.perf_counter()
        txt_probs = text_model.predict(text)[0]
        end_txt = time.perf_counter()
        
        # Fusion
        start_fuse = time.perf_counter()
        ensemble.fuse(img_probs.tolist(), txt_probs.tolist())
        end_fuse = time.perf_counter()
        
        end_total = time.perf_counter()
        
        image_latencies.append((end_img - start_img) * 1000)
        text_latencies.append((end_txt - start_txt) * 1000)
        fusion_latencies.append((end_fuse - start_fuse) * 1000)
        total_latencies.append((end_total - start_total) * 1000)
        
    # 5. Report & Save
    output_dir = 'outputs/performance'
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = []
    report_lines.append("# System Inference Speed Report")
    report_lines.append(f"*Measured on: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    report_lines.append(f"*Device: {DEVICE}*")
    report_lines.append("\n" + "="*50)
    report_lines.append("🚦 INFERENCE SPEED METRICS (ms)")
    report_lines.append("="*50)
    
    results = [
        ("Image (CNN+QNN)", image_latencies),
        ("Text (BioBERT)", text_latencies),
        ("Ensemble Fusion", fusion_latencies),
        ("Total Local Flow", total_latencies)
    ]
    
    metrics_data = {
        "device": str(DEVICE),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "components": {}
    }
    
    for name, data in results:
        comp_metrics = {
            "avg": float(np.mean(data)),
            "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95)),
            "min": float(np.min(data))
        }
        metrics_data["components"][name] = comp_metrics
        
        report_lines.append(f"\n### {name}:")
        report_lines.append(f"- **Average**: {comp_metrics['avg']:.2f} ms")
        report_lines.append(f"- **Median**:  {comp_metrics['median']:.2f} ms")
        report_lines.append(f"- **P95**:     {comp_metrics['p95']:.2f} ms")
        report_lines.append(f"- **Min**:     {comp_metrics['min']:.2f} ms")
        
    throughput = 1000 / np.mean(total_latencies)
    metrics_data["overall_throughput_ips"] = float(throughput)
    
    report_lines.append("\n" + "="*50)
    report_lines.append(f"**System Throughput: {throughput:.2f} inferences/sec**")
    report_lines.append("="*50)
    
    # Print to console
    print("\n".join(report_lines))
    
    # Save files
    with open(os.path.join(output_dir, 'inference_speed_report.md'), 'w', encoding='utf-8') as f:
        f.writelines([line + "\n" for line in report_lines])
        
    import json
    with open(os.path.join(output_dir, 'inference_speed_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"\n✅ Reports saved to {output_dir}")

if __name__ == "__main__":
    main()
