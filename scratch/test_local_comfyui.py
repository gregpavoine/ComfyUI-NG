import json
import urllib.request
import urllib.parse
import time

def test_local():
    # Simple T2I graph in ComfyUI API format
    prompt_dict = {
        "3": {
            "inputs": {
                "seed": 42,
                "steps": 4,
                "cfg": 3.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "ZImageTurbo/base model/gonzalomoZpop_v40.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": "A beautiful cute orange kitten playing with yarn",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "comfyng_test"
            },
            "class_type": "SaveImage"
        }
    }

    req_data = json.dumps({"prompt": prompt_dict}).encode('utf-8')
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    
    print("Queueing prompt on port 8188...")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode('utf-8'))
        prompt_id = res["prompt_id"]
        print(f"Prompt queued! prompt_id: {prompt_id}")
    except Exception as e:
        print(f"Failed to queue prompt: {e}")
        return

    # Poll history
    print("Polling history...")
    for i in range(30):
        time.sleep(1.0)
        history_url = f"http://127.0.0.1:8188/history/{prompt_id}"
        try:
            with urllib.request.urlopen(history_url, timeout=5) as response:
                hist = json.loads(response.read().decode('utf-8'))
            if prompt_id in hist:
                print("Execution finished!")
                outputs = hist[prompt_id]["outputs"]
                print("Outputs:", outputs)
                # Find SaveImage node (9) output
                node_output = outputs.get("9")
                if node_output and "images" in node_output:
                    img_info = node_output["images"][0]
                    filename = img_info["filename"]
                    print(f"Output image name: {filename}")
                    # Download image
                    view_url = f"http://127.0.0.1:8188/view?filename={filename}&subfolder=&type=output"
                    with urllib.request.urlopen(view_url, timeout=5) as view_res:
                        img_bytes = view_res.read()
                    print(f"Downloaded image bytes length: {len(img_bytes)}")
                break
            else:
                print(f"[{i+1}s] Still running...")
        except Exception as e:
            print(f"Polling error: {e}")

if __name__ == "__main__":
    test_local()
