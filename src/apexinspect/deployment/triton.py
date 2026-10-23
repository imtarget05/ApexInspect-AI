"""
Triton Inference Server deployment utilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import subprocess
import time
import requests
import json


class TritonDeployer:
    """Deploy models to Triton Inference Server."""
    
    def __init__(
        self,
        model_repository: str = "/models",
        triton_image: str = "nvcr.io/nvidia/tritonserver:24.04-py3",
        http_port: int = 8000,
        grpc_port: int = 8001,
        metrics_port: int = 8002,
    ):
        self.model_repository = Path(model_repository)
        self.triton_image = triton_image
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.metrics_port = metrics_port
        self.container_name = "apexinspect-triton"
    
    def create_model_repository(self, models: Dict[str, Dict]) -> None:
        """
        Create Triton model repository structure.
        
        Args:
            models: Dict of model_name -> {version_policy, input_shape, etc.}
        """
        self.model_repository.mkdir(parents=True, exist_ok=True)
        
        for model_name, config in models.items():
            model_dir = self.model_repository / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create version directory
            version_dir = model_dir / "1"
            version_dir.mkdir(exist_ok=True)
            
            # Copy model file
            if "model_path" in config:
                import shutil
                src = Path(config["model_path"])
                dst = version_dir / src.name
                shutil.copy2(src, dst)
            
            # Create config.pbtxt
            self._write_config_pbtxt(model_dir, config)
    
    def _write_config_pbtxt(self, model_dir: Path, config: Dict):
        """Write Triton config.pbtxt."""
        config_content = f"""
name: "{config.get('name', 'model')}"
backend: "{config.get('backend', 'onnxruntime')}"
max_batch_size: {config.get('max_batch_size', 8)}
input {{
  name: "{config.get('input_name', 'images')}"
  data_type: TYPE_FP32
  dims: {config.get('input_shape', [3, 640, 640])}
}}
output {{
  name: "{config.get('output_name', 'output')}"
  data_type: TYPE_FP32
  dims: {config.get('output_shape', [-1, 6])}
}}
instance_group {{
  count: {config.get('instance_count', 1)}
  kind: KIND_GPU
}}
dynamic_batching {{
  max_queue_delay_microseconds: {config.get('max_queue_delay', 10000)}
}}
optimization_policy {{
  batch_size: {config.get('batch_size', 8)}
}}
"""
        (model_dir / "config.pbtxt").write_text(config_content)
    
    def start_server(self, detach: bool = True, gpus: str = "all") -> subprocess.Popen:
        """Start Triton server in Docker."""
        cmd = [
            "docker", "run",
            "--rm" if not detach else "",
            "--name", self.container_name,
            "-p", f"{self.http_port}:8000",
            "-p", f"{self.grpc_port}:8001",
            "-p", f"{self.metrics_port}:8002",
            "-v", f"{self.model_repository.absolute()}:/models",
            "--gpus", gpus,
            self.triton_image,
            "tritonserver",
            "--model-repository=/models",
            "--strict-model-config=false",
            "--log-verbose=1",
        ]
        
        cmd = [c for c in cmd if c]  # Remove empty strings
        
        print(f"Starting Triton server: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait for server to be ready
        self.wait_ready()
        
        return process
    
    def wait_ready(self, timeout: int = 60) -> bool:
        """Wait for Triton to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"http://localhost:{self.http_port}/v2/health/ready", timeout=2)
                if resp.status_code == 200:
                    print("Triton server is ready!")
                    return True
            except:
                pass
            time.sleep(2)
        
        raise TimeoutError("Triton server failed to start")
    
    def stop_server(self):
        """Stop Triton server."""
        subprocess.run(["docker", "stop", self.container_name], capture_output=True)
    
    def test_model(self, model_name: str, input_data: Dict) -> Dict:
        """Test model inference via HTTP."""
        url = f"http://localhost:{self.http_port}/v2/models/{model_name}/infer"
        
        payload = {
            "inputs": [
                {
                    "name": k,
                    "shape": list(v.shape),
                    "datatype": "FP32",
                    "data": v.flatten().tolist(),
                }
                for k, v in input_data.items()
            ],
            "outputs": [{"name": "output"}],
        }
        
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    
    def get_model_metadata(self, model_name: str) -> Dict:
        """Get model metadata."""
        url = f"http://localhost:{self.http_port}/v2/models/{model_name}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    
    def get_model_stats(self, model_name: str) -> Dict:
        """Get model inference statistics."""
        url = f"http://localhost:{self.http_port}/v2/models/{model_name}/stats"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()


def create_triton_config(
    model_name: str,
    backend: str = "onnxruntime",
    input_name: str = "images",
    input_shape: List[int] = [3, 640, 640],
    output_name: str = "output",
    output_shape: List[int] = [-1, 6],
    max_batch_size: int = 8,
    instance_count: int = 1,
    max_queue_delay: int = 10000,
    backend_params: Optional[Dict] = None,
) -> str:
    """Generate Triton config.pbtxt content."""
    
    config = f"""
name: "{model_name}"
backend: "{backend}"
max_batch_size: {max_batch_size}
input {{
  name: "{input_name}"
  data_type: TYPE_FP32
  dims: {input_shape}
}}
output {{
  name: "{output_name}"
  data_type: TYPE_FP32
  dims: {output_shape}
}}
instance_group {{
  count: {instance_count}
  kind: KIND_GPU
}}
dynamic_batching {{
  max_queue_delay_microseconds: {max_queue_delay}
}}
"""
    
    if backend_params:
        config += "parameters {\n"
        for key, value in backend_params.items():
            config += f'  key: "{key}"\n  value: {{ string_value: "{value}" }}\n'
        config += "}\n"
    
    return config


def deploy_with_docker_compose(
    compose_path: Union[str, Path],
    model_repository: str,
    services: Optional[List[str]] = None,
) -> subprocess.Popen:
    """Deploy with docker-compose."""
    compose_path = Path(compose_path)
    compose_path.parent.mkdir(parents=True, exist_ok=True)
    
    compose_content = f"""version: '3.8'

services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.04-py3
    container_name: apexinspect-triton
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - {model_repository}:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: tritonserver --model-repository=/models --strict-model-config=false
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-data:
"""
    
    compose_path.write_text(compose_content)
    
    cmd = ["docker-compose", "-f", str(compose_path), "up"]
    if services:
        cmd.extend(services)
    if False:  # detach
        cmd.append("-d")
    
    return subprocess.Popen(cmd)


__all__ = [
    "TritonDeployer",
    "create_triton_config",
    "deploy_with_docker_compose",
]