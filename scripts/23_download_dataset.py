"""
Download PlantDisease dataset from Kaggle and verify class compatibility
"""
import kagglehub
import os
import sys
import pickle

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

def get_model_classes():
    """Get the classes our model expects"""
    data_path = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
    
    print("📦 Loading model's expected classes...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    label_encoder = data['label_encoder']
    classes = label_encoder.classes_
    
    print(f"\n✅ Model expects {len(classes)} classes:")
    for i, cls in enumerate(classes):
        print(f"   {i}. {cls}")
    
    return classes

def download_dataset():
    """Download PlantDisease dataset from Kaggle"""
    print("\n" + "="*80)
    print("DOWNLOADING PLANTDISEASE DATASET FROM KAGGLE")
    print("="*80)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("emmarex/plantdisease")
        
        print(f"\n✅ Dataset downloaded successfully!")
        print(f"📁 Path to dataset files: {path}")
        
        # List contents
        print(f"\n📋 Dataset structure:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files)-5} more files')
            if level > 2:  # Don't go too deep
                break
        
        return path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed kagglehub: pip install kagglehub")
        print("   2. Set up Kaggle API credentials")
        return None

def check_class_compatibility(dataset_path, model_classes):
    """Check if dataset classes match model classes"""
    print("\n" + "="*80)
    print("CHECKING CLASS COMPATIBILITY")
    print("="*80)
    
    # Typical PlantDisease dataset structure: dataset_path/PlantVillage/class_name/*.jpg
    plant_village_path = os.path.join(dataset_path, "PlantVillage")
    
    if os.path.exists(plant_village_path):
        dataset_classes = sorted([d for d in os.listdir(plant_village_path) 
                                 if os.path.isdir(os.path.join(plant_village_path, d))])
    else:
        # Try other common structures
        dataset_classes = sorted([d for d in os.listdir(dataset_path) 
                                 if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"\n📊 Dataset has {len(dataset_classes)} classes:")
    for i, cls in enumerate(dataset_classes[:20]):  # Show first 20
        print(f"   {i}. {cls}")
    if len(dataset_classes) > 20:
        print(f"   ... and {len(dataset_classes)-20} more classes")
    
    # Check overlap
    model_set = set(model_classes)
    dataset_set = set(dataset_classes)
    
    overlap = model_set & dataset_set
    missing_in_dataset = model_set - dataset_set
    extra_in_dataset = dataset_set - model_set
    
    print(f"\n🔍 Compatibility Analysis:")
    print(f"   ✅ Classes in both: {len(overlap)}")
    print(f"   ⚠️  In model but not dataset: {len(missing_in_dataset)}")
    print(f"   ℹ️  In dataset but not model: {len(extra_in_dataset)}")
    
    if missing_in_dataset:
        print(f"\n⚠️  Missing classes:")
        for cls in list(missing_in_dataset)[:10]:
            print(f"      - {cls}")
    
    if len(overlap) >= len(model_classes) * 0.8:
        print(f"\n✅ Good compatibility! {len(overlap)}/{len(model_classes)} classes match")
    else:
        print(f"\n⚠️  Limited compatibility: only {len(overlap)}/{len(model_classes)} classes match")
    
    return dataset_classes

def main():
    print("\n" + "="*80)
    print("PLANTDISEASE DATASET SETUP")
    print("="*80)
    
    # Get model's expected classes
    model_classes = get_model_classes()
    
    # Download dataset
    dataset_path = download_dataset()
    
    if dataset_path:
        # Check compatibility
        dataset_classes = check_class_compatibility(dataset_path, model_classes)
        
        print(f"\n💾 Dataset is ready at: {dataset_path}")
        print(f"\n📝 Next steps:")
        print(f"   1. Update evaluation script to use this dataset path")
        print(f"   2. Map dataset classes to model classes if needed")
        print(f"   3. Run ensemble evaluation with real images")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
