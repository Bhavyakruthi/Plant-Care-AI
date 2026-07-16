"""
Script to generate and save Grad-CAM heatmap as a viewable image file.
Run this to see the XAI visualizations saved to disk.
"""
import requests
import base64
import os
from pathlib import Path

def save_gradcam_image():
    print("\n" + "="*80)
    print("GRAD-CAM HEATMAP VIEWER")
    print("="*80)
    
    # Configuration
    url = "http://localhost:8000/diagnose"
    test_image = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\CNN+QNN codes\Healthy (11).jpg"
    output_dir = Path("output_gradcam")
    output_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📡 Analyzing: {os.path.basename(test_image)}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (os.path.basename(test_image), f)}
            response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend response received.\n")
            
            # Display diagnosis
            print(f"🎯 DIAGNOSIS: {data['disease']}")
            print(f"📊 CONFIDENCE: {data['confidence']}")
            print(f"🔬 IMAGE CONTRIBUTION: {data['image_contribution']}")
            print(f"📝 TEXT CONTRIBUTION: {data['text_contribution']}\n")
            
            # Display Gemini description
            if 'description' in data:
                desc = data['description']
                print(f"📝 GEMINI ANALYSIS:")
                print(f"   {desc[:200]}{'...' if len(desc) > 200 else ''}\n")
            
            # Display text importance
            if 'importance' in data and data['importance']:
                print("🔍 KEY DIAGNOSTIC FEATURES:")
                for feature, score in sorted(data['importance'].items(), key=lambda x: x[1], reverse=True):
                    print(f"   • {feature:<12} : {score:.4f}")
                print()
            
            # Save Grad-CAM heatmap
            if 'heatmap' in data and data['heatmap']:
                heatmap_b64 = data['heatmap']
                heatmap_bytes = base64.b64decode(heatmap_b64)
                
                output_path = output_dir / f"gradcam_{data['disease']}.png"
                with open(output_path, 'wb') as f:
                    f.write(heatmap_bytes)
                
                print(f"✅ GRAD-CAM HEATMAP SAVED TO:")
                print(f"   {output_path.absolute()}")
                print(f"   Size: {len(heatmap_bytes):,} bytes\n")
                
                # Auto-open the image
                try:
                    os.startfile(output_path.absolute())
                    print("🖼️  Opening image viewer...")
                except Exception as e:
                    print(f"   (Could not auto-open: {e})")
            else:
                print("❌ No heatmap in response")
        else:
            print(f"❌ Backend error: {response.status_code}")
            print(f"   {response.text}")
    
    except requests.exceptions.Timeout:
        print("❌ Request timeout (>60s). Gemini API may be slow.")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    save_gradcam_image()
