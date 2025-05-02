import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
import gradio as gr


# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


azure_openai = AzureChatOpenAI(
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-08-01-preview",  # Specify API version
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment="gpt-4o",
        model="gpt-4o" # Or another model you have deployed
    )

# Initialize Azure OpenAI client for image generation
azure_image_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY_DALL_E"],  # Make sure this is set to your Azure OpenAI API key
    api_version="2024-02-01",  # Update to the version specified in the documentation
    azure_endpoint="https://open-ai-key-align.openai.azure.com/"  # Use the endpoint from your documentation
)

#system_message = "You are a helpful assistant"
system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."

def generate_image(prompt, size="1024x1024"):
    """Generate an image using Azure OpenAI's DALL-E model."""
    try:
        response = azure_image_client.images.generate(
            model="dall-e-3",  # This is the deployment name from your documentation
            prompt=prompt,
            n=1  # Generate one image
        )
        # Extract the URL using the approach shown in the documentation
        return response.data[0].url
    except Exception as e:
        print(f"Error generating image: {e}")
        return None



def chat(message, history):
    # Check if user is requesting an image
    if message.lower().startswith("generate image:") or message.lower().startswith("create image:"):
        image_prompt = message.split(":", 1)[1].strip()
        image_url = generate_image(image_prompt)
        
        if image_url:
            return f"![Generated Image]({image_url})\n\nHere's the image you requested. Is there anything else you'd like to see?"
        else:
            return "I'm sorry, I couldn't generate that image. Please try a different description."
    
    # Format messages in LangChain format
    formatted_history = []
    
    # Add system message
    formatted_history.append({
        "role": "system", 
        "content": system_message
    })
    
    # Add past conversation turns if any
    if history:
        for entry in history:
            # Gradio ChatInterface passes history as a list of [user_message, assistant_message] pairs
            user_msg, ai_msg = entry
            formatted_history.append({"role": "user", "content": user_msg})
            formatted_history.append({"role": "assistant", "content": ai_msg})
    
    # Add the new user message
    formatted_history.append({"role": "user", "content": message})
    
    print("History is:")
    print(history)
    print("Formatted messages:")
    print(formatted_history)

    # Use the LangChain AzureChatOpenAI client for streaming
    try:
        response = ""
        for chunk in azure_openai.stream(formatted_history):
            if chunk and hasattr(chunk, 'content') and chunk.content:
                response += chunk.content
                yield response
        
        # If no content was generated, return a fallback message
        if not response:
            yield "I'm sorry, I couldn't process your request. Please try again."
            
    except Exception as e:
        print(f"Error in chat function: {e}")
        yield f"I apologize, but I encountered an error: {str(e)}"
            
gr.ChatInterface(fn=chat, type="messages").launch()

