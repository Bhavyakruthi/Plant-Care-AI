"""
Load model using ZIP-based loading for PyTorch directory archives
"""
import torch
import os
import zipfile
import tempfile
import shutil

# The directory that PyTorch saved as a ZIP structure
model_dir = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\models\image\cnn_qnn_best.pt"
output_path = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\models\image\cnn_qnn_weights.pth"

print("="*80)
print("CONVERTING PYTORCH ZIP ARCHIVE TO REGULAR .PTH FILE")
print("="*80)

try:
    # PyTorch's ZIP format can be loaded using torch.jit.load or special methods
    # Let's try using internal torch serialization
    from torch.serialization import _legacy_load
    from torch._utils import _import_dotted_name
    
    # Try loading as a package
    print(f"\nAttempting to load from: {model_dir}")
    
    # Method 1: Try using torch.package
    try:
        from torch.package import PackageImporter
        importer = PackageImporter(model_dir)
        checkpoint = importer.load_pickle('archive', 'data.pkl')
        print("✅ Loaded using PackageImporter!")
    except Exception as e1:
        print(f"PackageImporter failed: {e1}")
        
        # Method 2: Manual ZIP loading
        print("\nTrying manual ZIP extraction...")
        import pickle
        
        # Read the data.pkl file directly
        data_pkl_path = os.path.join(model_dir, 'data.pkl')
        if os.path.exists(data_pkl_path):
            # Create custom unpickler that can resolve the tensor references
            class TorchUnpickler(pickle.Unpickler):
                def __init__(self, file, **kwargs):
                    super().__init__(file, **kwargs)
                    self.model_dir = model_dir
                    
                def persistent_load(self, pid):
                    # Load tensors from the data directory
                    if isinstance(pid, tuple) and len(pid) > 0:
                        typename = pid[0]
                        if typename == 'storage':
                            storage_type, key, location, size = pid[1:]
                            # Load from data/key file
                            dtype = storage_type.dtype
                            data_file = os.path.join(self.model_dir, 'data', str(key))
                            
                            if os.path.exists(data_file):
                                # Read raw tensor data
                                import numpy as np
                                with open(data_file, 'rb') as f:
                                    data = np.fromfile(f, dtype=np.uint8)
                                # Convert to torch tensor
                                storage = torch.ByteStorage.from_buffer(data.tobytes())
                                return storage
                    return None
            
            with open(data_pkl_path, 'rb') as f:
                unpickler = TorchUnpickler(f)
                checkpoint = unpickler.load()
            
            print("✅ Loaded using custom unpickler!")
        else:
            raise FileNotFoundError(f"data.pkl not found in {model_dir}")
    
    # Now save as regular .pth
    print(f"\nCheckpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    torch.save(checkpoint, output_path)
    print(f"\n✅ Successfully saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
