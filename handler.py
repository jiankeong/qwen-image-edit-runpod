"""
RunPod Hub handler declaration.

The Docker image is based on runpod/worker-comfyui, which already contains the
production ComfyUI Serverless handler and startup process. This file is kept in
the repository for RunPod Hub/GitHub validation and documents the same handler
contract.
"""
import runpod

def handler(job):
    job_input = job.get("input") or {}
    if "workflow" not in job_input:
        return {"error": "Missing 'workflow' parameter"}
    return {
        "error": (
            "Repository validation handler only. "
            "The built Docker image uses worker-comfyui's production handler."
        )
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
