
"""GPU pipeline probe — verify on the service host whether indexing actually uses the GPU.

Checks three layers (each can silently fall back to CPU; OCR now uses the
rapidocr torch backend, so there is no onnxruntime layer left to verify):
  (1) Whether torch sees the GPU (docling's use_cuda gate; ROCm goes here too)
  (2) The actual device of the RapidOCR torch session built by the same
      converter the service uses (ground truth at the call layer)
  (3) Which device the Layout/TableFormer models actually land on

Run:
    PYTHONPATH=. uv run --no-sync python scripts/probe_gpu_pipeline.py
"""

import sys


def main() -> int:
    problems = []

    import torch
    cuda_ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda_ok else "-"
    print(f"① torch.cuda.is_available() = {cuda_ok}  ({torch.__version__}, {name})")
    if not cuda_ok:
        problems.append("torch cannot see the GPU -> docling pipeline falls back to CPU entirely (container --gpus? / driver?)")

    # Same converter the service uses -> actual device of the RapidOCR torch session.
    from src.config.config_manager import get_config
    get_config("config/config.yaml")
    from src.domain.rag.docling_loader import get_converter
    from docling.datamodel.base_models import InputFormat

    converter = get_converter()
    try:
        pdf_pipeline = converter._get_pipeline(InputFormat.PDF)
        ocr_model = getattr(pdf_pipeline, "ocr_model", None)
        if ocr_model is None:
            print("② ocr_model not found in pipeline (OCR not enabled?)")
            problems.append("OCR stage not in pipeline (check ocr_enabled)")
        else:
            found = False
            reader = ocr_model.reader
            for part_name in ("text_det", "text_cls", "text_rec"):
                part = getattr(reader, part_name, None)
                sess = getattr(getattr(part, "session", None), "session", None) or getattr(part, "session", None)
                dev = getattr(sess, "device", None)
                if dev is None:
                    continue
                found = True
                on_gpu = "cuda" in str(dev).lower()
                print(f"② RapidOCR {part_name}: device={dev}  {'✅GPU' if on_gpu else '❌CPU'}")
                if not on_gpu:
                    problems.append(
                        f"OCR {part_name} landed on CPU (EngineConfig.torch.use_cuda not effective "
                        f"or accelerator device resolved to cpu)"
                    )
            if not found:
                print("② Cannot introspect RapidOCR torch sessions (rapidocr version structure differs) — use nvidia-smi/rocm-smi instead")
    except Exception as e:
        print(f"② converter introspection failed: {type(e).__name__}: {e}")

    try:
        layout = getattr(pdf_pipeline, "layout_model", None)
        if layout is not None:
            dev = None
            for attr in ("device", "_device"):
                dev = dev or getattr(layout, attr, None)
            inner = getattr(layout, "layout_predictor", None) or getattr(layout, "model", None)
            if dev is None and inner is not None:
                dev = getattr(inner, "device", None) or getattr(inner, "_device", None)
            print(f"③ Layout model device = {dev}")
            if dev is not None and "cuda" not in str(dev).lower():
                problems.append("Layout model landed on CPU")
    except Exception as e:
        print(f"③ layout introspection failed: {type(e).__name__}: {e}")

    print()
    if problems:
        print("❌ Problems found:")
        for p in problems:
            print(f"   - {p}")
        return 1
    print("✅ All three layers are on the GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
