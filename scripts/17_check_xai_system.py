import requests
import os
import base64

def check_xai_status():
    print("\n" + "="*80)
    print("XAI PIPELINE VERIFICATION")
    print("="*80)
    
    url = "http://localhost:8000/diagnose"
    test_image = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\CNN+QNN codes\Healthy (11).jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found at: {test_image}")
        return
    print(f"📡 Sending '{test_image}' to Multimodal AI...")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image, f)}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend response received.")
            
            # Check XAI Fields
            if "heatmap" in data and data["heatmap"]:
                print(f"✅ Grad-CAM Heatmap: RECEIVED ({len(data['heatmap'])} bytes)")
            else:
                print("❌ Grad-CAM Heatmap: MISSING")
                
            if "importance" in data and data["importance"]:
                print("✅ Text Importance: RECEIVED")
                for feature, score in data["importance"].items():
                    print(f"   - {feature:<10} : {score:.4f}")
            else:
                print("❌ Text Importance: MISSING")
                
            print(f"\n🎯 Diagnosis: {data['disease']}")
            print(f"📝 Gemini Description (Snippet): {data['description'][:60]}...")
            
        else:
            print(f"❌ Backend error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("   Make sure 'uvicorn main:app --reload' is running in the backend folder.")

    print("\n" + "="*80)

if __name__ == "__main__":
    check_xai_status()
