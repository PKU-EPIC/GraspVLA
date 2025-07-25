# FIXME: import urchin before others, otherwise segfault, unknown reason

import os
if 'DEBUG_PORT' in os.environ:
    import debugpy
    debugpy.listen(int(os.environ['DEBUG_PORT']))
    print(f'waiting for debugger to attach...')
    debugpy.wait_for_client()

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--port", type=str, required=True)
arg_parser.add_argument("--path", type=str, required=True)
arg_parser.add_argument("--compile", action="store_true")


import PIL
import io
import os
from typing import List
import zmq
import pickle
import time
import numpy as np
from tqdm import tqdm
from vla_network.model.vla import VLAAgent
from vla_network.data_preprocessing.prompt import COT_PROMPT
import torch
torch.autograd.set_grad_enabled(False)


def interpolate_delta_actions(delta_actions, n):
    """
    Interpolate m delta_actions to m*n delta_actions.

    actions: list of actions, each action is (delta x, delta y, delta z, delta roll, delta pitch, delta yaw, gripper open/close).
    """
    import transforms3d as t3d
    ret = []
    for delta_action in delta_actions:
        xyzs = 1 / n * np.array([delta_action[:3]]*n)
        axangle_ax, axangle_angle = t3d.euler.euler2axangle(*delta_action[3:6])
        eulers = [t3d.euler.axangle2euler(axangle_ax, axangle_angle / n)]*n
        grippers = np.array([[0.]] * (n-1) + [[delta_action[-1]]])  # 0 for no change of gripper state
        ret.extend(np.concatenate([xyzs, eulers, grippers], axis=-1))
    return ret


def infer_single_sample(vla_model: VLAAgent, sample: dict):
    input_data = []
    if sample.get('compressed', False):
        for key in ['front_view_image', 'side_view_image']:
            decompressed_image_array = []
            for compressed_image in sample[key]:
                decompressed_image_array.append(np.array(PIL.Image.open(io.BytesIO(compressed_image))))
            sample[key] = decompressed_image_array
        sample['compressed'] = False
    proprio_array = np.array([sample['proprio_array'][-4], sample['proprio_array'][-1]])
    proprio_array[:, -1] = (proprio_array[:, -1] + 1) / 2
    input_data.append({
        'env_id': 0,
        'text': COT_PROMPT(sample['text']),
        'proprio_array': proprio_array,
        'front_view_image': [sample['front_view_image'][-1]],
        'side_view_image': [sample['side_view_image'][-1]],
    })
    results = vla_model(input_data) # the model recieve a list of input samples
    result = results[0]  # only one sample
    action = result['action']
    # Quantize last dimension of action to 0, 1 using 0.4, 0.6 as bin boundaries
    last_dim = action[:, -1]
    last_dim = np.where(last_dim < 0.4, -1, np.where(last_dim > 0.6, 1, 0))
    action = np.concatenate([action[:, :-1], last_dim[:, None]], axis=-1)
    action = interpolate_delta_actions(action, 2)
    debug = {}
    if 'goal' in result:
        debug['pose'] = result['goal']
    if 'bbox' in result:
        debug['bbox'] = result['bbox']
    return {
        'result': action,
        'env_id': 0,
        'debug': debug,
    }


def warmup(vla_model: VLAAgent):
    SAMPLES = [
        {
            'text': 'pick up elephant',
            'front_view_image': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'side_view_image': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
        },
        {
            'text': 'pick up toy large elephant',
            'front_view_image': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'side_view_image': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
        },
        {
            'text': 'pick up toy car',
            'front_view_image': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'side_view_image': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
        },
    ]
    NUM_TESTS = 5
    print('warming up...')
    for i in tqdm(range(NUM_TESTS)):
        ret = infer_single_sample(vla_model, SAMPLES[i%len(SAMPLES)])
    print('check the latency after warm up:')
    for i in tqdm(range(NUM_TESTS)):
        ret = infer_single_sample(vla_model, SAMPLES[i%len(SAMPLES)])


def main():
    args = arg_parser.parse_args()
    vla_model = VLAAgent(args.path, compile=args.compile)
    
    vla_model.preprocessor.config.robot_rep = "identity"

    assert vla_model.data_cfg.action_rel_len == 0

    warmup(vla_model)

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{args.port}")

    print(f"Started server on port {args.port}")

    requests = []

    while True:
       # run inference if data is ready
        if (len(requests) > 0):
            client_id, data_received = requests[0]
            
            tbegin = time.time()
            print(f'start processing a request')
            result = infer_single_sample(vla_model, data_received)
            tend = time.time()
            print(f'finished a request in {tend-tbegin:.3f}s')

            socket.send_multipart([
                client_id,
                b'',
                pickle.dumps({
                    'info': 'success',
                    'env_id': result['env_id'],
                    'result': result['result'],
                    'debug': result['debug'],
                })
            ])

            requests = requests[1:]

        # try getting new sample
        try:
            client_id, empty, data = socket.recv_multipart(zmq.DONTWAIT)
            data = pickle.loads(data)
            requests.append((client_id, data))
        except zmq.Again:
            pass


if __name__ == "__main__":
    main()
