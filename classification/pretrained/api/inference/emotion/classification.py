from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoConfig,
)
import os
import json
from pathlib import Path

try:
    import safetensors.torch as safe_torch
    from safetensors import safe_open as st_safe_open
except Exception:
    safe_torch = None
    st_safe_open = None
import torch
import torch.nn.functional as F
import numpy as np

# Disable torch.compile/dynamo to avoid meta tracing/device issues on CPU
# If getting error similar to the following:
# ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
# Run pip install accelerate

try:
    import torch._dynamo as _dynamo  # type: ignore

    _dynamo.disable()
except Exception:
    pass


class Infer:

    def __init__(
        self, model="dima806/facial_emotions_image_detection", debug: bool = True
    ):

        self.debug = debug
        self.model = model
        self.model_type = "image-classification"

        try:
            self._init_model_and_processor()
        except Exception as e:
            self.print_debug(f"Error initializing model/processor: {e}")
            raise

    def print_debug(self, message: str):

        if self.debug:

            print(f"[DEBUG] {message}")

    def update_pipeline(self):
        try:
            # Rebuild model/processor on CPU
            self._init_model_and_processor()
        except Exception as e:
            self.print_debug(f"Error updating model/processor: {str(e)}")

    def change_model(self, new_model: str):

        self.model = new_model
        self.update_pipeline()

    def set_image(self, image):

        self.image = image

    def _assert_no_meta(self):
        try:

            def is_meta_tensor(t: torch.Tensor) -> bool:
                try:
                    if getattr(t, "is_meta", False):
                        return True
                except Exception:
                    pass
                try:
                    return getattr(t, "device", torch.device("cpu")).type == "meta"
                except Exception:
                    return False

            meta_params = []
            for name, p in self._model.named_parameters():
                if isinstance(p, torch.Tensor) and is_meta_tensor(p):
                    meta_params.append(name)
            meta_buffers = []
            for name, b in self._model.named_buffers():
                if isinstance(b, torch.Tensor) and is_meta_tensor(b):
                    meta_buffers.append(name)

            if meta_params or meta_buffers:
                self.print_debug(
                    f"Meta parameters: {meta_params[:10]}{'...' if len(meta_params)>10 else ''}"
                )
                self.print_debug(
                    f"Meta buffers: {meta_buffers[:10]}{'...' if len(meta_buffers)>10 else ''}"
                )
                # Try to materialize once
                self.print_debug("Attempting to materialize meta tensors on CPU...")
                self._materialize_meta_tensors()
                # Re-check
                meta_params2 = []
                for name, p in self._model.named_parameters():
                    if isinstance(p, torch.Tensor) and is_meta_tensor(p):
                        meta_params2.append(name)
                meta_buffers2 = []
                for name, b in self._model.named_buffers():
                    if isinstance(b, torch.Tensor) and is_meta_tensor(b):
                        meta_buffers2.append(name)
                if meta_params2 or meta_buffers2:
                    self.print_debug(
                        f"Still meta after materialization. Params: {meta_params2[:10]} Buffers: {meta_buffers2[:10]}"
                    )
                    # Do not raise; proceed and let forward complain only if those tensors are used
        except Exception as e:
            self.print_debug(f"Meta tensor check failed: {e}")
            raise

    def _model_device(self):
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def predict(self):

        self.print_debug("Running prediction...")

        try:
            if not hasattr(self, "image") or self.image is None:
                self.print_debug("No image set for prediction")
                return []

            self.print_debug(
                f"Image type: {type(self.image)}, size: {self.image.size if hasattr(self.image, 'size') else 'unknown'}"
            )

            inputs = self._processor(images=self.image, return_tensors="pt")
            device = self._model_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits  # [1, num_labels]
                probs = F.softmax(logits[0], dim=0).to("cpu").numpy()

            id2label = self._model.config.id2label
            top_idx = np.argsort(-probs)[:5].tolist()
            results = [
                {"label": str(id2label.get(i, str(i))), "score": float(probs[i])}
                for i in top_idx
            ]

            self.print_debug(f"Raw prediction results: {results}")
            return results

        except Exception as e:
            self.print_debug(f"Error during prediction: {str(e)}")
            import traceback

            self.print_debug(f"Traceback: {traceback.format_exc()}")
            return []

    def _materialize_meta_tensors(self) -> None:
        """Replace any remaining meta parameters/buffers with CPU tensors (zeros)."""

        def _set_by_name(module, name: str, value, is_param: bool):
            parts = name.split(".")
            sub = module
            for p in parts[:-1]:
                sub = getattr(sub, p)
            leaf = parts[-1]
            if is_param:
                setattr(sub, leaf, torch.nn.Parameter(value, requires_grad=False))
            else:
                # Prefer register_buffer if available
                try:
                    sub.register_buffer(leaf, value, persistent=True)
                except Exception:
                    sub._buffers[leaf] = value

        with torch.no_grad():
            # Parameters
            for name, p in list(self._model.named_parameters()):
                is_meta = False
                try:
                    is_meta = getattr(p, "is_meta", False) or (p.device.type == "meta")
                except Exception:
                    is_meta = getattr(p, "is_meta", False)
                if is_meta:
                    cpu_tensor = torch.zeros(
                        tuple(p.shape), dtype=p.dtype, device="cpu"
                    )
                    _set_by_name(self._model, name, cpu_tensor, is_param=True)
            # Buffers
            for name, b in list(self._model.named_buffers()):
                is_meta = False
                try:
                    is_meta = getattr(b, "is_meta", False) or (b.device.type == "meta")
                except Exception:
                    is_meta = getattr(b, "is_meta", False)
                if is_meta:
                    cpu_tensor = torch.zeros(
                        tuple(b.shape), dtype=b.dtype, device="cpu"
                    )
                    _set_by_name(self._model, name, cpu_tensor, is_param=False)

    def _init_model_and_processor(self) -> None:
        """Initialize model and processor avoiding meta tensors.

        Strategy:
        - Try standard from_pretrained (CPU, no device_map, no low-mem tricks).
        - If any parameter/buffer remains on meta, rebuild from config and
          load state dict manually via huggingface_hub (bin or safetensors).
        """
        last_err = None

        # 1) Standard path
        try:
            self.print_debug("Loading model via from_pretrained (standard path)...")
            self._model = AutoModelForImageClassification.from_pretrained(
                self.model,
                low_cpu_mem_usage=False,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self._model.eval()
            self._processor = AutoImageProcessor.from_pretrained(
                self.model, trust_remote_code=True
            )
            # Initialize any lazy params by a dry forward
            self._dry_initialize()
            self._assert_no_meta()
            self.print_debug("Model loaded (standard path).")
            return
        except Exception as e:
            last_err = e
            self.print_debug(
                f"Standard load failed or meta detected; falling back. Reason: {e}"
            )

        # 2) Fallback: manual config + state dict (supports sharded weights)
        try:
            self.print_debug("Loading config and building model from config...")
            cfg = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
            model = AutoModelForImageClassification.from_config(
                cfg, trust_remote_code=True
            )

            # Try safetensors/bin; support single-file and sharded via index
            state_dict = None
            try:
                from huggingface_hub import (
                    snapshot_download,  # type: ignore
                )
            except Exception as e:
                raise RuntimeError(
                    "huggingface_hub is required for fallback loading. Install it: pip install huggingface_hub"
                ) from e

            repo_dir = snapshot_download(repo_id=self.model)
            repo = Path(repo_dir)

            # Single files
            single_safe = None
            for fname in ["model.safetensors", "pytorch_model.safetensors"]:
                p = repo / fname
                if p.exists():
                    single_safe = str(p)
                    break

            single_bin = None
            for fname in ["pytorch_model.bin"]:
                p = repo / fname
                if p.exists():
                    single_bin = str(p)
                    break

            # Sharded index
            idx_safe = repo / "model.safetensors.index.json"
            idx_bin = repo / "pytorch_model.bin.index.json"

            if single_safe and safe_torch is not None:
                self.print_debug(f"Loading single safetensors file: {single_safe}")
                state_dict = safe_torch.load_file(single_safe, device="cpu")
            elif single_bin:
                self.print_debug(f"Loading single bin file: {single_bin}")
                state_dict = torch.load(single_bin, map_location="cpu")
            elif idx_safe.exists() and st_safe_open is not None:
                self.print_debug("Loading sharded safetensors via index...")
                with open(idx_safe, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                weight_map: dict = idx_data.get("weight_map", {})
                state_dict = {}
                # Group keys by shard file
                file_to_keys: dict[str, list[str]] = {}
                for k, shard in weight_map.items():
                    file_to_keys.setdefault(shard, []).append(k)
                for shard, keys in file_to_keys.items():
                    shard_path = repo / shard
                    self.print_debug(f"Reading shard: {shard}")
                    with st_safe_open(
                        str(shard_path), framework="pt", device="cpu"
                    ) as f:
                        for k in keys:
                            try:
                                state_dict[k] = f.get_tensor(k)
                            except Exception:
                                pass
            elif idx_bin.exists():
                self.print_debug("Loading sharded bin weights via index...")
                with open(idx_bin, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                weight_map: dict = idx_data.get("weight_map", {})
                state_dict = {}
                # Group keys by shard file
                file_to_keys: dict[str, list[str]] = {}
                for k, shard in weight_map.items():
                    file_to_keys.setdefault(shard, []).append(k)
                for shard, _ in file_to_keys.items():
                    shard_path = repo / shard
                    self.print_debug(f"Reading shard: {shard}")
                    try:
                        shard_sd = torch.load(str(shard_path), map_location="cpu")
                        state_dict.update(
                            {k: v for k, v in shard_sd.items() if k in weight_map}
                        )
                    except Exception:
                        pass
            else:
                raise RuntimeError(
                    "Could not locate model weights (.safetensors/.bin, single or sharded) in snapshot."
                )

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                self.print_debug(f"Missing keys when loading state dict: {missing}")
            if unexpected:
                self.print_debug(
                    f"Unexpected keys when loading state dict: {unexpected}"
                )

            self._model = model.eval()
            self._processor = AutoImageProcessor.from_pretrained(
                self.model, trust_remote_code=True
            )
            # Initialize any lazy params by a dry forward
            self._dry_initialize()
            self._assert_no_meta()
            self.print_debug("Model loaded via manual state-dict fallback.")
            return
        except Exception as e:
            last_err = e
            self.print_debug(
                f"Fallback (config + state dict) also failed: {type(e).__name__}: {e}"
            )
            raise last_err

    def get_numeric_scores(self, results):
        try:
            if not results or not isinstance(results, list):
                self.print_debug(f"Invalid results format: {results}")
                return {}

            numeric_scores = {}
            # Map common label variants to our canonical set
            label_map = {
                "angry": "anger",
                "anger": "anger",
                "happiness": "happy",
                "happy": "happy",
                "sadness": "sad",
                "sad": "sad",
                "surprised": "surprise",
                "surprise": "surprise",
                "disgust": "disgust",
                "neutral": "neutral",
                "fear": "fear",
            }
            for item in results:
                if isinstance(item, dict) and "label" in item and "score" in item:
                    try:
                        score = float(item["score"]) * 10
                        if not np.isnan(score) and score >= 0:
                            raw_label = str(item["label"]).lower()
                            label = label_map.get(raw_label, raw_label)
                            numeric_scores[label] = score
                    except (ValueError, TypeError) as e:
                        self.print_debug(f"Error processing score for {item}: {e}")

            self.print_debug(f"Processed numeric scores: {numeric_scores}")
            return numeric_scores

        except Exception as e:
            self.print_debug(f"Error getting numeric scores: {str(e)}")
            import traceback

            self.print_debug(f"Traceback: {traceback.format_exc()}")
            return {}

    def _dry_initialize(self):
        """Run a tiny CPU-only forward pass to initialize any lazy/meta tensors.

        Creates a dummy 224x224 RGB image, runs it through the processor and model
        once under no_grad. This helps materialize weights in lazy modules.
        """
        try:
            from PIL import Image as _Image

            dummy = _Image.new("RGB", (224, 224), (0, 0, 0))
            inputs = self._processor(images=dummy, return_tensors="pt")
            device = torch.device("cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = self._model(**inputs)
            self.print_debug("Dry initialization forward completed.")
        except Exception as e:
            # Not fatal; continue loading
            self.print_debug(f"Dry initialization skipped due to error: {e}")
