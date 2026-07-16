"""
Convert the ZIP-archive model to a single .pth file
"""
import torch
import os

# Path to the ZIP-archive model (directory structure)
model_dir = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\models\image\cnn_qnn_best.pt"
output_path = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\models\image\cnn_qnn_best_fixed.pth"

print("Converting ZIP-archive model to single .pth file...")

try:
    # Load as ZIP archive
    checkpoint = torch.load(model_dir, map_location='cpu', weights_only=False)
    
    print(f"✅ Successfully loaded checkpoint")
    print(f"Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    # Save as regular .pth file
    torch.save(checkpoint, output_path)
    print(f"✅ Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
