import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import webbrowser
from pathlib import Path
import requests
import time

# Load environment variables
load_dotenv()

def test_image_generation():
    # Initialize Azure OpenAI client for image generation
    azure_image_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY_DALL_E"],
        api_version="2024-02-01",
        azure_endpoint="https://open-ai-key-align.openai.azure.com/"
    )
    
    # Get a prompt from the user
    prompt = input("Enter an image prompt: ")
    
    print(f"Generating image for: '{prompt}'...")
    
    try:
        # Generate the image
        response = azure_image_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1
        )
        
        # Extract the URL
        image_url = response.data[0].url
        print(f"Image generated successfully!")
        print(f"Image URL: {image_url}")
        
        # Ask if user wants to save the image
        save_image = input("Would you like to save this image locally? (y/n): ").lower()
        
        if save_image == 'y':
            # Create an images folder if it doesn't exist
            image_dir = Path("./images")
            image_dir.mkdir(exist_ok=True)
            
            # Download the image
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                # Create a filename based on timestamp
                filename = f"image_{int(time.time())}.png"
                filepath = image_dir / filename
                
                # Save the image
                with open(filepath, 'wb') as f:
                    f.write(img_response.content)
                print(f"Image saved to {filepath}")
            else:
                print(f"Failed to download image. Status code: {img_response.status_code}")
        
        # Ask if user wants to open the image in browser
        open_browser = input("Would you like to view this image in your browser? (y/n): ").lower()
        
        if open_browser == 'y':
            webbrowser.open(image_url)
            
        return True
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return False

if __name__ == "__main__":
    # Check if the API key is set
    if "AZURE_OPENAI_API_KEY_DALL_E" not in os.environ or not os.environ["AZURE_OPENAI_API_KEY_DALL_E"]:
        print("ERROR: AZURE_OPENAI_API_KEY_DALL_E environment variable is not set.")
        print("Please set this variable in your .env file or export it directly.")
        exit(1)
    
    print("Starting DALL-E image generation test...")
    success = test_image_generation()
    
    if success:
        print("Test completed successfully!")
        
        # Ask if the user wants to generate another image
        while input("Generate another image? (y/n): ").lower() == 'y':
            test_image_generation()
            
        print("Goodbye!")
    else:
        print("Test failed. Please check your API key and endpoint.")