import urllib.request
import json

def get_checkpoints():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=5) as response:
            info = json.loads(response.read().decode('utf-8'))
        ckpt_loader = info.get("CheckpointLoaderSimple")
        if ckpt_loader:
            ckpt_names = ckpt_loader["input"]["required"]["ckpt_name"][0]
            print("Checkpoint options on port 8188:", ckpt_names)
        else:
            print("CheckpointLoaderSimple not found in object_info.")
    except Exception as e:
        print(f"Error fetching object_info: {e}")

if __name__ == "__main__":
    get_checkpoints()
