from urllib import request
import zmq
from PIL import Image
import io
import cv2
from matplotlib import pyplot as plt
import numpy as np
from termcolor import colored
import os

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--port", type=str, default="6666")

def validate_server(host: str = "127.0.0.1", port: int = 6666, timeout: int = 5) -> bool:
    """
    Validate that the server is running and returns a valid dict.
    
    Args:
        host: Server hostname
        port: Server port
        timeout: Timeout in seconds
        
    Returns:
        True if server returns valid dict, False otherwise
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
    
    try:
        socket.connect(f"tcp://{host}:{port}")
        
        # Create test data matching agent.py format
        mock_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mock_proprio = [np.random.randn(7) for _ in range(4)]
        
        test_data = {
            'front_view_image': [mock_image],
            'side_view_image': [mock_image],
            'proprio_array': mock_proprio,
            'text': 'Validation test instruction',
        }
        
        socket.send_pyobj(test_data)
        response = socket.recv_pyobj()
        
        # Check if response is a valid dict
        if not isinstance(response, dict):
            print(f"✗ Server returned {type(response)}, expected dict")
            return False
            
        print(colored(f"✓ Server at {host}:{port} returned valid dict", 'green'))
        return True
        
    except zmq.Again:
        print(colored(f"✗ Server at {host}:{port} timeout after {timeout}s", 'red'))
        return False
    except Exception as e:
        print(colored(f"✗ Error connecting to server at {host}:{port}: {e}", 'red'))
        return False
    finally:
        socket.close()
        context.term()


def rename_request_keys(request):
    if 'image_array' in request:
        request['front_view_image'] = request.pop('image_array')
    if 'image_wrist_array' in request:
        request['side_view_image'] = request.pop('image_wrist_array')


def visualize_response(request, response, vis=False):
    bbox = response['debug']['bbox']

    if request['compressed']:
        front_image = Image.open(io.BytesIO(request['front_view_image'][0]))
        side_image = Image.open(io.BytesIO(request['side_view_image'][0]))
    else:
        front_image = request['front_view_image'][0]
        side_image = request['side_view_image'][0]
    front_image = np.array(front_image)
    front_bbox = bbox[0]
    resized_bbox = (front_bbox / 224 * 256).astype(int) # hack: the order of image and bbox is different
    cv2.rectangle(front_image, (resized_bbox[0], resized_bbox[1]), (resized_bbox[2], resized_bbox[3]), (0, 255, 0), 2)

    side_image = np.array(side_image)
    side_bbox = bbox[1]
    resized_bbox = (side_bbox / 224 * 256).astype(int) # hack: the order of image and bbox is different
    cv2.rectangle(side_image, (resized_bbox[0], resized_bbox[1]), (resized_bbox[2], resized_bbox[3]), (0, 255, 0), 2)

    merged_image = np.concatenate((front_image, side_image), axis=1)
    if vis:
        plt.imshow(merged_image)
    return merged_image
 

def main():

    args = arg_parser.parse_args()

    if (validate_server(port=args.port) == True):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://127.0.0.1:{args.port}")

        trial_caption = 'trial-20250507120350'
        data = np.load("visualization/trial-20250507120350_data.npy", allow_pickle=True).item()
        print(f"Task: {data['request']['text']}")

        fig = plt.figure(figsize=(6, 2))
        plt.suptitle(f"Task: {data['request']['text']}", fontsize=16)

        request = data['request']
        response = data['response']

        rename_request_keys(request)

        # show the result of the original model
        left_image = visualize_response(request, response)

        try:
            socket.send_pyobj(request)
            new_response = socket.recv_pyobj()
        except Exception as e:
            print(f"Socket communication failed: {e}")

        # show the result of the current model
        right_image = visualize_response(request, new_response)

        # our model result
        plt.subplot(1, 2, 1)
        plt.imshow(left_image)
        plt.title(f"Our Result")
        plt.axis('off')

        # current model result
        plt.subplot(1, 2, 2)
        plt.imshow(right_image)
        plt.title(f"Your Result")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        os.makedirs("visualization", exist_ok=True)
        plt.savefig(f"visualization/{trial_caption}_visualization.png", dpi=200)
        print(f"Saved figure as \"visualization/{trial_caption}_visualization.png\".")
        socket.close()
        context.term()

    else:
        print(f"Please make sure the model server is running at tcp://127.0.0.1:{args.port}.")
        return

if __name__ == "__main__":
    main()