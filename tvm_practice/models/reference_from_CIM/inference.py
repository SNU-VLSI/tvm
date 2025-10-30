from deploy_modules import ResNet8IMCFlow, ResNet8
import argparse
import torch
import torch.nn as nn
import sys
import os
import logging
import datetime

# parent directory added to load utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import eval, data_loader


def inference(adjust_factors, state_dict, verbose=False):
  args = argparse.Namespace()
  args.dataset = 'cifar10'
  args.data = '../data/cifar10'
  args.train_batch = 256
  args.test_batch = 256
  args.valid_size = 0.1
  args.workers = 4
  args.gpu_id = '0'
  args.rank = 0 if verbose else -1
  args.distributed = False
  args.augment = False
  args.class_split = None
  args.per_class = None
  args.perf_sample = False
  args.perf_list = None
  args.dali = False
  args.zca = False
  args.whist = False
  args.num_classes = 10

  _, _, valid_loader, test_loader = data_loader.set_data_loader(args)
  criterion = nn.CrossEntropyLoss()

  model_fp = ResNet8(state_dict).cuda()
  model_imcflow = ResNet8IMCFlow(state_dict, adjust_factors).cuda()

  top1_fp = {'valid': 0, 'test': 0}
  top5_fp = {'valid': 0, 'test': 0}
  loss_fp = {'valid': None, 'test': None}

  top1_int16 = {'valid': 0, 'test': 0}
  top5_int16 = {'valid': 0, 'test': 0}
  loss_int16 = {'valid': None, 'test': None}

  loss_fp['valid'], top1_fp['valid'], top5_fp['valid'] = eval.test(valid_loader, model_fp, criterion, 0, args)
  loss_fp['test'], top1_fp['test'], top5_fp['test'] = eval.test(test_loader, model_fp, criterion, 0, args)

  loss_int16['valid'], top1_int16['valid'], top5_int16['valid'] = eval.test(
    valid_loader, model_imcflow, criterion, 0, args)
  loss_int16['test'], top1_int16['test'], top5_int16['test'] = eval.test(test_loader, model_imcflow, criterion, 0, args)

  return loss_fp, top1_fp, top5_fp, loss_int16, top1_int16, top5_int16, model_imcflow


def main():
  logging.basicConfig(level=logging.ERROR)
  checkpoint_path = "A4W4+PS6/2025-Sep-24-01-20-40"
  state_dict = torch.load(f"../models_checkpoint/{checkpoint_path}/checkpoint.pth.tar")
  adjust_factors = {
      'x_f_1': 36.0,
      'bn1_f_1': 72.0,
      'bn2_f_1': 36.0,
      'x_f_2': 36.0,  # needs to be the same with bn2_f_1
      'bn1_f_2': 150.0,
      'bn2_f_2': 15.0,
      'x_f_3': 10.0,
      'bn1_f_3': 50.0,
      'bn2_f_3': 500.0,
  }

  loss_fp, top1_fp, top5_fp, loss_int16, top1_int16, top5_int16, model_imcflow = inference(adjust_factors, state_dict, verbose=True)

  print(f"Valid loss fp: {loss_fp['valid']:<10.6f} Valid top1: {top1_fp['valid']:<7.4f} Valid top5: {top5_fp['valid']:<7.4f}")
  print(f"Test loss fp: {loss_fp['test']:<10.6f} Test top1: {top1_fp['test']:<7.4f} Test top5: {top5_fp['test']:<7.4f}")
  print(f"Valid loss int16: {loss_int16['valid']:<10.6f} Valid top1: {top1_int16['valid']:<7.4f} Valid top5: {top5_int16['valid']:<7.4f}")
  print(f"Test loss int16: {loss_int16['test']:<10.6f} Test top1: {top1_int16['test']:<7.4f} Test top5: {top5_int16['test']:<7.4f}")

  # Save the model_imcflow state_dict including all registered buffers
  imcflow_state_dict = model_imcflow.state_dict()

  # Create a comprehensive state dict that includes the original checkpoint info
  save_dict = {
      'state_dict': imcflow_state_dict,
      'adjust_factors': adjust_factors,
      'original_checkpoint': state_dict,  # Keep reference to original checkpoint
      'model_type': 'ResNet8IMCFlow',
      'inference_results': {
          'loss_fp': loss_fp,
          'top1_fp': top1_fp,
          'top5_fp': top5_fp,
          'loss_int16': loss_int16,
          'top1_int16': top1_int16,
          'top5_int16': top5_int16
      }
  }

  # Save to file with timestamp
  timestamp = datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
  save_path = f"../models_checkpoint/{checkpoint_path}/imcflow/{timestamp}/checkpoint.pth.tar"

  # Create directory if it doesn't exist
  os.makedirs(os.path.dirname(save_path), exist_ok=True)

  torch.save(save_dict, save_path)
  print(f"\nSaved IMCFlow model state_dict to: {save_path}")
  print(f"State dict keys: {list(imcflow_state_dict.keys())}")


if __name__ == "__main__":
  main()
