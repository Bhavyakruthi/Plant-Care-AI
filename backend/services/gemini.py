import google.generativeai as genai
import os
from PIL import Image

class GeminiService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def generate_description(self, image_path: str) -> str:
        """
        Generates a detailed botanical description of plant disease symptoms from an image.
        """
        try:
            with Image.open(image_path) as img:
                img.load() # Load into memory
                prompt = """
                Analyze this plant leaf image for disease diagnosis. 
                Provide a detailed technical description focusing on:
                1. Morphology: Leaf shape, edge type, and overall structure.
                2. Lesions: Color (e.g., chlorotic yellow, necrotic brown), shape (circular, irregular), and distribution (marginal, interveinal).
                3. Symptoms: Presence of mold, wilting, curling, or spots.
                
                CRITICAL INSTRUCTION: DO NOT mention the name of the plant or leaf (e.g., do not say 'Tomato', 'Corn', 'Grape', etc.). 
                Describe only the physical symptoms and visual features as seen in the image.
                
                Use standard botanical and plant pathological terminology. 
                Keep the description concise but comprehensive (around 100-150 words).
                """
                
                response = self.model.generate_content([prompt, img])
                return response.text
        except Exception as e:
            return f"Error generating description: {str(e)}"

    def generate_description_bytes(self, img_bytes: bytes) -> str:
        import io
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                img.load()
                prompt = """
                Analyze this plant leaf image for disease diagnosis. 
                Provide a detailed technical description focusing on:
                1. Morphology: Leaf shape, edge type, and overall structure.
                2. Lesions: Color (e.g., chlorotic yellow, necrotic brown), shape (circular, irregular), and distribution (marginal, interveinal).
                3. Symptoms: Presence of mold, wilting, curling, or spots.
                
                CRITICAL INSTRUCTION: DO NOT mention the name of the plant or leaf (e.g., do not say 'Tomato', 'Corn', 'Grape', etc.). 
                Describe only the physical symptoms and visual features as seen in the image.
                
                Use standard botanical and plant pathological terminology. 
                Keep the description concise but comprehensive (around 100-150 words).
                """
                
                response = self.model.generate_content([prompt, img])
                return response.text
        except Exception as e:
            return f"Error generating description: {str(e)}"

# Placeholder for testing
if __name__ == "__main__":
    # Example usage (would require API key)
    # service = GeminiService(api_key="YOUR_KEY")
    # print(service.generate_description("test.jpg"))
    pass
