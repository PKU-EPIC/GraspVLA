# FIXME: import urchin before others, otherwise segfault, unknown reason
from urchin import URDF, Collision, Sphere, Geometry # type: ignore

import os
if 'DEBUG_PORT' in os.environ:
    import debugpy
    debugpy.listen(int(os.environ['DEBUG_PORT']))
    print(f'waiting for debugger to attach...')
    debugpy.wait_for_client()

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--port", type=str, required=True)
arg_parser.add_argument("--batching-delay", type=float, default=80)
arg_parser.add_argument("--batch-size", type=int, default=1)
arg_parser.add_argument("--path", type=str, required=True)
arg_parser.add_argument("--compile", action="store_true")


import PIL
import io
from typing import List
import zmq
import pickle
import time
import numpy as np
from tqdm import tqdm
from vla_network.model.vla.agent import VLAAgent
from vla_network.datasample.vla import VLASample
from vla_network.utils.logger import log
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


def batch_process(vla_model: VLAAgent, batch: List[dict]):
    for sample in batch:
        if sample.get('compressed', False):
            for key in ['image_array', 'image_wrist_array']:
                decompressed_image_array = []
                for compressed_image in sample[key]:
                    decompressed_image_array.append(np.array(PIL.Image.open(io.BytesIO(compressed_image))))
                sample[key] = decompressed_image_array
            sample['compressed'] = False

    assert vla_model.config.model.image_keys == ["left", "right"]  # TODO: make it universal
    dt_steps = round(vla_model.config.model.dt / 0.1)  # TODO: make it universal
    frame_indices = list(range(-(vla_model.config.model.proprio_steps-1)*dt_steps-1, 0, dt_steps))
    input_batch = []
    for sample in batch:
        input_batch.append(VLASample(
            dataset_name="agent",
            embodiment='epiclab_franka',
            frame=0,
            instruction=sample['text'],
            images=dict(
                left=[PIL.Image.fromarray(sample['image_wrist_array'][i]) for i in frame_indices[-vla_model.config.model.image_steps:]],
                right=[PIL.Image.fromarray(sample['image_array'][i]) for i in frame_indices[-vla_model.config.model.image_steps:]],
            ),
            proprio=np.array([sample['proprio_array'][i] for i in frame_indices]),
        ))
    results: List[VLASample] = vla_model(input_batch)
    ret = []
    for result, input_sample in zip(results, batch):
        action = result.action
        # Quantize last dimension of action to 0, 1 using 0.4, 0.6 as bin boundaries
        last_dim = action[:, -1]
        last_dim = np.where(last_dim < -0.5, -1, np.where(last_dim > 0.5, 1, 0))
        action = np.concatenate([action[:, :-1], last_dim[:, None]], axis=-1)
        action = interpolate_delta_actions(action, dt_steps)
        debug = {}
        if result.goal is not None:
            debug['pose'] = (result.goal[:3], result.goal[3:6])
        if result.bboxs is not None:
            debug['bbox'] = [result.bboxs[k][-1] for k in vla_model.config.model.image_keys]
        ret.append({
            'result': action,
            'env_id': input_sample['env_id'],
            'debug': debug,
        })
    return ret


def warmup(vla_model: VLAAgent):
    SAMPLES = [
        {
            'text': 'pick up elephant',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
            'traj_metadata': None,
            'env_id': 1,
        },
        {
            'text': 'pick up toy large elephant',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
            'traj_metadata': None,
            'env_id': 2,
        },
        {
            'text': 'pick up toy car',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
            'traj_metadata': None,
            'env_id': 3,
        },
    ]
    NUM_TESTS = 5
    print('warming up...')
    for i in tqdm(range(NUM_TESTS)):
        ret = batch_process(vla_model, [SAMPLES[i%len(SAMPLES)]])
    print('check the latency after warm up:')
    for i in tqdm(range(NUM_TESTS)):
        ret = batch_process(vla_model, [SAMPLES[i%len(SAMPLES)]])


def main():
    args = arg_parser.parse_args()
    vla_model = VLAAgent(args.path, compile=args.compile)

    warmup(vla_model)

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{args.port}")

    requests = []
    first_arrive_time = None

    log.info('start serving')
    while True:
       # run inference if batch is ready
        current_time = time.time() * 1000
        if (len(requests) >= args.batch_size or 
            ((first_arrive_time is not None) and 
             (current_time - first_arrive_time > args.batching_delay) and 
             len(requests) > 0)):
            data_num = min(args.batch_size, len(requests))
            client_ids, data_batch = zip(*requests[:data_num])

            tbegin = time.time()
            log.info(f'start processing {len(requests)} requests')
            results = batch_process(vla_model, data_batch)
            tend = time.time()
            log.info(f'finished {len(requests)} requests in {tend - tbegin:.3f}s')

            for client_id, result in zip(client_ids, results):
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

            requests = requests[data_num:]
            if len(requests) == 0:
                first_arrive_time = None

        # try getting new sample
        try:
            client_id, empty, data = socket.recv_multipart(zmq.DONTWAIT)
            if len(requests) == 0:
                first_arrive_time = time.time() * 1000

            data = pickle.loads(data)
            requests.append((client_id, data))
        except zmq.Again:
            pass


if __name__ == "__main__":
    main()
