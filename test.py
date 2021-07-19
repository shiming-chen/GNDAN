import torch
from eval_func import eval_zs_gzsl
from model import GNDAN
from dataset import UNIDataloader
import argparse
import json


def run_test(config):
    # dataset
    dataloader = UNIDataloader(config)
    # model
    model = GNDAN(config)
    # load parameters
    model_dict = model.state_dict()
    saved_dict = torch.load(config.saved_model)
    saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)
    model.to(config.device)
    # evaluation
    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(config, dataloader, model)
    print('acc_unseen={:.3f}, acc_seen={:.3f}, H={:.3f}, acc_zs={:.3f}'.format(
        acc_novel, acc_seen, H, acc_zs))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test_CUB.json')
    config = parser.parse_args()
    with open(config.config, 'r') as f:
        config.__dict__ = json.load(f)
    run_test(config)
