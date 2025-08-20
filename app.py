import requests
import json
import gradio as gr
from requests.exceptions import ConnectionError, RequestException
import time

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json'
}

history = []

def generate_response(prompt):
    if not prompt.strip():
        return "Please enter a valid prompt."
    
    history.append(prompt)
    final_prompt = "\n".join(history)

    data = {
        "model": "codellama:7b",  # Updated model here
        "prompt": final_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
        
        if response.status_code == 200:
            response_data = response.json()
            actual_response = response_data.get('response', 'No response received')
            return actual_response
        else:
            error_msg = f"Server error ({response.status_code}): {response.text}"
            print(error_msg)
            return error_msg
            
    except ConnectionError:
        error_msg = """
        Connection Error: Cannot connect to Ollama server.
        
        Please make sure:
        1. Ollama is installed
        2. Ollama server is running (`ollama serve`)
        3. The 'codellama:7b' model is available (`ollama pull codellama:7b`)
        """
        print(error_msg)
        return error_msg
        
    except RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        print(error_msg)
        return error_msg
        
    except json.JSONDecodeError:
        error_msg = "Invalid JSON response from server"
        print(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return error_msg

def clear_history():
    global history
    history = []
    return "History cleared!"

# Gradio UI
with gr.Blocks() as interface:
    gr.Markdown("# 🤖 CodeLlama 7B Chat Interface")
    gr.Markdown("Ensure Ollama is running and model 'codellama:7b' is pulled.")

    with gr.Row():
        input_box = gr.Textbox(
            lines=4, 
            placeholder="Enter your prompt here...",
            label="Prompt"
        )

    with gr.Row():
        submit_btn = gr.Button("Generate Response", variant="primary")
        clear_btn = gr.Button("Clear History", variant="secondary")

    output_box = gr.Textbox(
        lines=10, 
        label="Response",
        interactive=False
    )

    submit_btn.click(
        fn=generate_response,
        inputs=input_box,
        outputs=output_box
    )

    clear_btn.click(
        fn=clear_history,
        inputs=None,
        outputs=output_box
    )

if __name__ == "__main__":
    interface.launch()
