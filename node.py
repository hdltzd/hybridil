import argparse
import time
import json
import numpy as np
import torch

import zmq
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from deployment.aubo_hybridil import (
    decode_numpy_request,
    encode_numpy_response,
)


def main(cfg):
    model_path = cfg['model_path']
    port = cfg['port']

    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=model_path)

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()

    # read rollout settings
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    while True:
        print('========================       wating for obs data...       ========================')
        request_parts = socket.recv_multipart()
        print("======================== received data, start processing... ========================")
        numpy_protocol = len(request_parts) > 1
        if numpy_protocol:
            name, data = decode_numpy_request(request_parts)
        else:
            data = json.loads(request_parts[0].decode("utf-8"))
            name = data["name"]
        pointcloud = np.asarray(data["pointcloud"])
        robot0_eef_pos = np.asarray(data["robot0_eef_pos"])
        robot0_eef_quat = np.asarray(data["robot0_eef_quat"])
        robot0_eef_wrench = np.asarray(data["robot0_eef_wrench"])
        obs = dict(pointcloud=pointcloud,
                   robot0_eef_pos=robot0_eef_pos,
                   robot0_eef_quat=robot0_eef_quat,
                   robot0_eef_wrench=robot0_eef_wrench)
        t0 = time.time()
        act_seq = policy.get_all_action(ob=obs)
        print("========================       inference time: {}           ========================".format(time.time() - t0))
        act_seq = np.array(act_seq)
        print("========================  process done, send data back...   ========================")
        if numpy_protocol:
            socket.send_multipart(encode_numpy_response(name, act_seq))
        else:
            output_data = {
                "name": name,
                "action": act_seq.tolist()
            }
            socket.send_string(json.dumps(output_data))
        print("========================        send data back done          ========================")
        print(" ")
        print(" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="(optional) path of trained model to run. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="(optional) port number to bind the server. Only needs to be provided if --config is not provided",
    )

    args = parser.parse_args()
    if args.config is not None:
        cfg = json.load(open(args.config, 'r'))
    else:
        cfg = {
            "model_path": args.model_path,
            "port": args.port
        }
    main(cfg)
