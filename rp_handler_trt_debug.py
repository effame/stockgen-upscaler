"""Debug handler - just echoes back input"""
import runpod, json

def handler(job):
    return {"status": "ok", "received": job.get("input", {})}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
