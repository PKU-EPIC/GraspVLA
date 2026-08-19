import argparse
import copy
import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zmq


def _normalize_request(request: dict) -> dict:
    """Convert older request keys to the protocol expected by serve.py."""
    request = copy.deepcopy(request)
    if "front_view_image" in request and "image_array" not in request:
        request["image_array"] = request.pop("front_view_image")
    if "side_view_image" in request and "image_wrist_array" not in request:
        request["image_wrist_array"] = request.pop("side_view_image")

    request.setdefault("text", "pick up the object")
    request.setdefault("env_id", 0)
    request.setdefault("compressed", False)
    required = ("image_array", "image_wrist_array", "proprio_array")
    missing = [key for key in required if key not in request]
    if missing:
        raise KeyError(f"Request is missing required fields: {missing}")
    return request


def _mock_request() -> dict:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 48, (255, 255, 255), -1)
    proprio = np.zeros(7, dtype=np.float32)
    proprio[-1] = 1.0
    return {
        "text": "pick up the white circle",
        "image_array": [image.copy()],
        "image_wrist_array": [image.copy()],
        "proprio_array": [proprio.copy() for _ in range(4)],
        "env_id": 0,
        "compressed": False,
    }


def load_case(path: str | None) -> tuple[dict, dict | None]:
    if path is None:
        return _mock_request(), None

    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(data).__name__}")
    request = data.get("request", data)
    reference = data.get("response")
    return _normalize_request(request), reference


def request_server(
    request: dict,
    host: str,
    port: int,
    timeout: float,
) -> dict:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
    socket.connect(f"tcp://{host}:{port}")
    try:
        socket.send_pyobj(request)
        response = socket.recv_pyobj()
    except zmq.Again as exc:
        raise TimeoutError(
            f"Server at {host}:{port} did not respond within {timeout:g}s"
        ) from exc
    finally:
        socket.close()
        context.term()

    if not isinstance(response, dict):
        raise TypeError(
            f"Server returned {type(response).__name__}; expected a dict"
        )
    if response.get("info") != "success":
        raise RuntimeError(f"Server returned an unsuccessful response: {response}")
    return response


def _decode_image(value, compressed: bool) -> np.ndarray:
    if compressed or isinstance(value, (bytes, bytearray)):
        return np.array(Image.open(io.BytesIO(value)).convert("RGB"))
    return np.asarray(value).copy()


def _draw_bbox(image: np.ndarray, bbox) -> np.ndarray:
    if bbox is None:
        return image
    height, width = image.shape[:2]
    box = np.rint(np.asarray(bbox)).astype(int)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    cv2.rectangle(
        image,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        (0, 255, 0),
        2,
    )
    return image


def visualize_response(request: dict, response: dict) -> np.ndarray:
    compressed = bool(request.get("compressed", False))
    side = _decode_image(request["image_wrist_array"][-1], compressed)
    front = _decode_image(request["image_array"][-1], compressed)

    bboxes = response.get("debug", {}).get("bbox")
    if bboxes is not None and len(bboxes) >= 2:
        # serve.py emits bboxes in model image-key order: left(side), right(front).
        side = _draw_bbox(side, bboxes[0])
        front = _draw_bbox(front, bboxes[1])
    return np.concatenate((side, front), axis=1)


def save_comparison(
    request: dict,
    response: dict,
    reference: dict | None,
    output: str,
) -> None:
    panels = [visualize_response(request, response)]
    titles = ["Current model"]
    if reference is not None:
        panels.insert(0, visualize_response(request, reference))
        titles.insert(0, "Reference")

    figure, axes = plt.subplots(
        1, len(panels), figsize=(7 * len(panels), 4), squeeze=False
    )
    figure.suptitle(request["text"])
    for axis, panel, title in zip(axes[0], panels, titles):
        axis.imshow(panel)
        axis.set_title(f"{title}: side/wrist | front/main")
        axis.axis("off")
    figure.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a recorded or mock request to the VLA ZMQ server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6666)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--input",
        help="Optional .npy file containing a request or request/response pair.",
    )
    parser.add_argument(
        "--output",
        default="visualization/offline_test.png",
        help="Path for the rendered comparison image.",
    )
    args = parser.parse_args()

    request, reference = load_case(args.input)
    request = _normalize_request(request)
    print(f"Task: {request['text']}")
    response = request_server(
        request=request,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    save_comparison(request, response, reference, args.output)
    print(f"Server response received; visualization saved to {args.output}")


if __name__ == "__main__":
    main()
