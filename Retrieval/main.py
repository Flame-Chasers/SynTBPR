import datetime
import os
from pprint import pprint
import random
import time
from pathlib import Path

import torch

from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from options import get_args
from model.eva_clip import EVA_CLIP
import json

def run(config):
    pprint(config)

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']

    meters = {
        "loss": AverageMeter(),
        "nitc_loss": AverageMeter(),
        "ritc_loss": AverageMeter()
    }
    best_rank_1 = 0.0
    best_epoch = 0


    with open(config.model.config_path, "r", encoding="utf8") as f:
        model_cfg = json.load(f)

    if 'rope' in model_cfg.get('vision_cfg', {}):
        if model_cfg['vision_cfg']['rope']:
            os.environ['RoPE'] = "1"
    else:
        os.environ['RoPE'] = "0"

    model = EVA_CLIP(config, **model_cfg)
    model.to(config.device)

    model, load_result = load_checkpoint(model, config)
    print(load_result)

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=False)
    model_without_ddp = model.module if config.distributed else model
    
    # eval_result = test(model, dataloader['test_loader'], 77, config.device)
    # rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
    # print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,
    #                                                                                 top10=rank_10, mAP=map))
    # print(1 / 0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 现在将timestamp添加到文件名中
    filename = f'checkpoint_last_{timestamp}'
    save_path = os.path.join(config.model.saved_path, filename)
    os.makedirs(save_path, exist_ok=True)

    # schedule
    config.schedule.niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config)

    # optimizer
    optimizer = build_optimizer(config, model)

    # train
    it = 0
    # scaler = torch.amp.GradScaler('cuda')
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.schedule.epoch):
        print()
        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for i, batch in enumerate(train_loader):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']

            if epoch == 0:
                alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
            else:
                alpha = config.model.softlabel_ratio

            with torch.autocast(device_type='cuda'):
                ret = model(batch, alpha)
                loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['image'].shape[0]
            meters['loss'].update(loss.item(), batch_size)
            meters['nitc_loss'].update(ret.get('nitc_loss', 0), batch_size)
            meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:
                info_str = f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"
                # log loss
                for k, v in meters.items():
                    if v.val != 0:
                        info_str += f", {k}: {v.val:.4f}"
                info_str += f", Base Lr: {param_group['lr']:.2e}"
                print(info_str)

        if is_master():
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (i + 1)
            print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                  .format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch))

            eval_result = test(model_without_ddp, dataloader['test_loader'], 77, config.device)
            rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
            print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,
                                                                                              top10=rank_10, mAP=map))
            torch.cuda.empty_cache()
            if best_rank_1 < rank_1:
                best_rank_1 = rank_1
                best_epoch = epoch

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }
                torch.save(save_obj, os.path.join(config.model.saved_path, f'checkpoint_best_{timestamp}.pth'))

        save_obj = {
                        'model': model.state_dict()
                    }
        torch.save(save_obj, os.path.join(save_path,f"epoch{epoch}.pth"))
        print("save_path = ",save_path)
        print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")


if __name__ == '__main__':
    config_path = 'config/config.yaml'

    args = get_args()
    if args.real_data:
        config_path = 'config/real_data_config.yaml'
    config = parse_config(config_path)

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)

    init_distributed_mode(config)

    set_seed(config)

    run(config)
