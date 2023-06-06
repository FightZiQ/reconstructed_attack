import pandas as pd
import torchvision
import torch
from DP.DP_model import SGM
from DP_util import get_samples, get_clipping_bound, get_std, attack_cfg_default, transforms, data_cfg_default, compress_to_original
from DP import proposed
from breaching.analysis.metrics import cw_ssim, psnr_compute, total_variation, mse_compute
import minimal_example_robbing_the_fed
import sys
from breaching.cases.malicious_modifications.imprint import ImprintBlock
from index import index_set, flower_index_set
sys.path.append("..")


device = torch.device('cpu')


def grad_comparison(mode='original'):
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset_set = ['ImageNet']
    model = torchvision.models.resnet101
    record = pd.DataFrame(columns=['batch size', 'bin num', 'norm', 'value'])
    clipping_bound = get_clipping_bound('resnet', 'resnet101', False)
    if mode == 'original':
        for times in range(10):
            batch_size = 16
            fig_idx = torch.randint(0, 10000, (batch_size,))
            datapoint, labels = get_samples(fig_idx, device=device, dataset=dataset_set[0])
            for pretrained in (False,):
                m = model(pretrained=pretrained).to(device)
                loss = loss_fn(m(datapoint), labels)
                gradients = torch.autograd.grad(loss, m.parameters())
                grad_norm = []
                for grad in gradients:
                    grad_norm.append(torch.norm(grad) ** 2)
                grad_norm = sum(grad_norm) ** 0.5
                print(times, model.__name__, pretrained, batch_size, grad_norm)
                record.loc[len(record)] = [dataset_set[0], grad_norm.cpu().item(), 'original']
                m = None  # to release memory
    elif mode == 'DP':
        for bs in (16, 64):
            for times in range(10):
                fig_idx = torch.randint(0, 10000, (bs,))
                for dataset in dataset_set:
                    datapoint, labels = get_samples(fig_idx, device=device, dataset=dataset)
                    for bin_num in [64, 128, 256, 512, 1024]:
                        for compress_image in (True, ):
                            torch.cuda.empty_cache()
                            ac = attack_cfg_default()
                            ac.compress_image = compress_image
                            ac.num_bins = bin_num
                            dc = data_cfg_default(compress_image)
                            dc.model = model(pretrained=False)

                            m = SGM(data_cfg=dc, attack_cfg=attack_cfg_default()).to(device)
                            loss = loss_fn(m(datapoint), labels)
                            gradients = torch.autograd.grad(loss, m.parameters())

                            grad_norm = []
                            for grad in gradients:
                                grad_norm.append(torch.norm(grad) ** 2)
                            grad_norm = sum(grad_norm) ** 0.5

                            clipping_factor = grad_norm / clipping_bound
                            if grad_norm > clipping_bound:
                                for grad in gradients:
                                    grad /= clipping_factor  # clipping
                            value = abs(gradients[0][gradients[0].nonzero(as_tuple=True)]).mean()
                            print(bs, bin_num, times, grad_norm, value)
                            record.loc[len(record)] = [bs, bin_num, grad_norm.cpu().item(), value.cpu().item()]
    elif mode == 'robbing':
        record = pd.DataFrame(columns=['model', 'pretrained', 'batch_size', 'bin_num', 'compress_image', 'norm'])
        for times in range(10):
            for batch_size in batch_size_set:
                fig_idx = torch.randint(0, 50000, (batch_size,))
                datapoint, labels = get_samples(fig_idx, device=device, dataset=dataset)
                for bin_num in [64, 128, 256, 512]:
                    for model in model_set:
                        for pretrained in (False, ):
                            for compress_image in (False,):
                                torch.cuda.empty_cache()
                                ac = attack_cfg_default()
                                ac.compress_image = compress_image
                                ac.num_bins = bin_num
                                dc = data_cfg_default(compress_image)
                                dc.model = model(pretrained=pretrained)

                                model = torchvision.models.resnet101(pretrained=False)
                                model.eval()
                                loss_fn = torch.nn.CrossEntropyLoss()
                                # It will be modified maliciously:
                                block = ImprintBlock(dc.shape, num_bins=ac.num_bins, gain=1)
                                model = torch.nn.Sequential(block, model).to(device)
                                loss = loss_fn(model(datapoint), labels)
                                gradients = torch.autograd.grad(loss, model.parameters())
                                grad_norm = []
                                for grad in gradients:
                                    grad_norm.append(torch.norm(grad) ** 2)
                                grad_norm = sum(grad_norm) ** 0.5
                                print('times: {0} | batch size: {1} | bin num: {2} | model: {3} | '
                                      'pretrained: {4} | compress image: {5} | norm: {6}'.format(
                                       times, batch_size, bin_num, 'resnet101', pretrained, ac.compress_image,
                                       grad_norm))
                                record.loc[len(record)] = ['resnet101', pretrained, batch_size, bin_num,
                                                           ac.compress_image, grad_norm.cpu().item()]
                                torch.cuda.empty_cache()
    record.to_json(r'evaluation/norm/batch size norm.json')


def fig_quality():
    cb = get_clipping_bound('resnet', 'resnet101', False)
    dc = data_cfg_default(compress_image=True)
    ac = attack_cfg_default()
    record = pd.DataFrame(columns=['batch size', 'bin num', 'res num', 'separated res', 'separated input',
                                   'overlapped res', 'overlapped input', 'meaningless', 'accuracy', 'ssim',
                                   'psnr', 'mse'])
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        dc.batch_size = bs
        for rounds in range(5):
            # fig_idx = torch.randint(0, 10000, (dc.batch_size,))
            for bin_num in (64, 128, 256, 512, 1024):
                ac.num_bins = bin_num
                print('batch size: {0}, bins num: {1}, round {2}'.format(dc.batch_size, ac.num_bins, rounds))

                res = proposed.perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, clipping_bound=cb)

                accuracy = len(res[1]) / dc.batch_size
                if len(res[2]) == 0:
                    # no reconstructed samples
                    continue
                separated_input = torch.stack(res[2])
                separated_res = torch.stack(res[1])
                if dc.compress_image:
                    separated_res = compress_to_original(separated_res, separated_input.size())
                ssim = cw_ssim(separated_res, separated_input)[0]
                psnr = psnr_compute(separated_res, separated_input)[0]
                mse = mse_compute(separated_res, separated_input)
                record.loc[len(record)] = [dc.batch_size, ac.num_bins, *list(map(len, res)), accuracy, ssim, psnr,
                                           mse]

    record.to_json(r'performance/record proposed.json', default_handler=str)


def robbing_the_fed(mode='original'):
    record = pd.DataFrame(columns=['batch size', 'bin num', 'clipping bound', 'separated fig'])

    # get the clipping bound
    cb = get_clipping_bound('resnet', 'resnet101', False)
    res = minimal_example_robbing_the_fed.main(saving_path=r'evaluation\vision',
                                               clipping_bound=cb, save_fig=True, batch_size=16, use_dp=True)
    return
    for times in range(10):
        for batch_size in (2 ** i for i in range(3, 7)):
            fig_idx = torch.randint(0, 10000, (batch_size,))
            for bin_num in (64, 128, 256, 512):
                minimal_example_robbing_the_fed.attack_cfg_default.num_bins = bin_num
                res = minimal_example_robbing_the_fed.main(saving_path=r'intermediate results\robbing recovered data.pt',
                                                           clipping_bound=cb, fig_idx=fig_idx, save_fig=False)
                record.loc[len(record)] = [batch_size, bin_num, 'resnet101', res]
                print('times: {0}| batch size: {1}| bin num: {2}| res: {3}'.format(times, batch_size, bin_num, res))
    record.to_json(r'performance/robbing/robbing res DP.json')


def fig_quality_three_metrics(dataset='ImageNet'):
    cb = get_clipping_bound('resnet', 'resnet101', False)
    dc = data_cfg_default(compress_image=True)
    ac = attack_cfg_default()
    idx_set = index_set
    if dataset == 'Flowers102':
        idx_set = flower_index_set
    record = pd.DataFrame(columns=['batch size', 'bin num', 'res num', 'separated res', 'separated input',
                                   'overlapped res', 'overlapped input', 'meaningless', 'accuracy', 'ssim',
                                   'psnr', 'mse'])
    dc.batch_size = 16
    ac.num_bins = 1024
    for fig_idx in idx_set:
        for rounds in range(5):
            print('batch size: {0}, bins num: {1}, round {2}'.format(dc.batch_size, ac.num_bins, rounds))

            res = proposed.perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, clipping_bound=cb,
                                           fig_idx=fig_idx, dataset=dataset, use_dp=True)

            accuracy = len(res[1]) / dc.batch_size
            if len(res[2]) == 0:
                # no reconstructed samples
                continue

            separated_input = torch.stack(res[2])
            separated_input = separated_input * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

            separated_res = torch.stack(res[1])
            if dc.compress_image:
                separated_res = compress_to_original(separated_res, separated_input.size())
            separated_res = separated_res * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

            ssim = cw_ssim(separated_res, separated_input)[0]
            psnr = psnr_compute(separated_res, separated_input)[0]
            mse = mse_compute(separated_res, separated_input).cpu().item()
            record.loc[len(record)] = [dc.batch_size, ac.num_bins, *list(map(len, res)), accuracy, ssim, psnr,
                                       mse]
    record.to_json(r'evaluation/comparison/%s proposed compressed noise.json' % dataset.lower(), default_handler=str)


def value_comparison():
    device = torch.device('cuda')
    loss_fn = torch.nn.CrossEntropyLoss()
    model = torchvision.models.resnet101
    record = pd.DataFrame(columns=['compressed', 'clipping', 'value'])
    dataset = 'ImageNet'
    clipping_bound = get_clipping_bound('resnet', 'resnet101', False)
    for times in range(10):
        print(times)
        fig_idx = torch.randint(0, 10000, (16,))
        datapoint, labels = get_samples(fig_idx, device=device, dataset=dataset)
        for bin_num in [1024, ]:
            for compress_image in (True, False):
                for clipping in (True, False):
                    torch.cuda.empty_cache()
                    ac = attack_cfg_default()
                    ac.compress_image = compress_image
                    ac.num_bins = bin_num
                    dc = data_cfg_default(compress_image)
                    dc.model = model(pretrained=False)

                    m = SGM(data_cfg=dc, attack_cfg=attack_cfg_default()).to(device)
                    loss = loss_fn(m(datapoint), labels)
                    gradients = torch.autograd.grad(loss, m.parameters())

                    if clipping:
                        grad_w = []
                        for grad in gradients:
                            grad_w.append(torch.norm(grad) ** 2)
                        grad_norm = sum(grad_w) ** 0.5

                        clipping_factor = grad_norm / clipping_bound
                        if grad_norm > clipping_bound:
                            for grad in gradients:
                                grad /= clipping_factor  # clipping

                    value = abs(gradients[0][gradients[0].nonzero(as_tuple=True)]).mean()
                    record.loc[len(record)] = [compress_image, clipping, value.cpu().item()]
    record.to_json(r'evaluation\norm\value.json', default_handler=str)


def factor_to_performance():
    cb = get_clipping_bound('resnet', 'resnet101', False)
    dataset = 'ImageNet'
    record = pd.DataFrame(columns=['batch size', 'bin num', 'res num', 'separated res', 'separated input',
                                   'overlapped res', 'overlapped input', 'meaningless', 'accuracy', 'ssim',
                                   'psnr', 'mse', 'Compressed input'])
    for bs in [64, ]:
        for rounds in range(1):
            fig_idx = torch.randint(0, 10000, (bs,))
            for unit in [2, ]:
                # for unit in [1024, ]:
                for compress_image in (True, False):
                    dc = data_cfg_default(compress_image=compress_image)
                    ac = attack_cfg_default()
                    dc.batch_size = bs
                    ac.num_bins = unit
                    print('batch size: {0}, bins num: {1}, round {2}'.format(dc.batch_size, ac.num_bins, rounds))

                    res = proposed.perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, clipping_bound=cb,
                                                   dataset=dataset, use_dp=True, fig_idx=fig_idx)

                    accuracy = len(res[1]) / dc.batch_size
                    if len(res[2]) == 0:
                        # no reconstructed samples
                        continue

                    separated_input = torch.stack(res[2])
                    separated_input = separated_input * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

                    separated_res = torch.stack(res[1])
                    if dc.compress_image:
                        separated_res = compress_to_original(separated_res, separated_input.size())
                    separated_res = separated_res * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

                    ssim = cw_ssim(separated_res, separated_input)[0]
                    psnr = psnr_compute(separated_res, separated_input)[0]
                    mse = mse_compute(separated_res, separated_input).cpu().item()
                    record.loc[len(record)] = [dc.batch_size, ac.num_bins, *list(map(len, res)), accuracy, ssim, psnr,
                                               mse, compress_image]
        record.to_json(r'evaluation/factor/bin num.json', default_handler=str)


def privacy_setting(mode='epsilon'):
    if mode == 'epsilon':
        cb = get_clipping_bound('resnet', 'resnet101', False)
        dataset = 'ImageNet'
        record = pd.DataFrame(columns=['batch size', 'bin num', 'res num', 'separated res', 'separated input',
                                       'overlapped res', 'overlapped input', 'meaningless', 'accuracy', 'ssim',
                                       'psnr', 'mse', 'epsilon'])
        for bs in [16, 32, 64]:
            for rounds in range(10):
                fig_idx = torch.randint(0, 10000, (bs,))
                for epsilon in [50, 75, 100, 125, 150, 175, 200]:
                    dc = data_cfg_default(compress_image=True)
                    ac = attack_cfg_default()
                    dc.batch_size = bs
                    ac.num_bins = 1024
                    print('batch size: {0}, epsilon: {1}, round {2}'.format(dc.batch_size, epsilon, rounds))

                    res = proposed.perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, clipping_bound=cb,
                                                   dataset=dataset, use_dp=True, fig_idx=fig_idx, epsilon=epsilon)
                    accuracy = len(res[1]) / dc.batch_size
                    if len(res[2]) == 0:
                        # no reconstructed samples
                        continue

                    separated_input = torch.stack(res[2])
                    separated_input = separated_input * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

                    separated_res = torch.stack(res[1])
                    if dc.compress_image:
                        separated_res = compress_to_original(separated_res, separated_input.size())
                    separated_res = separated_res * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

                    ssim = cw_ssim(separated_res, separated_input)[0]
                    psnr = psnr_compute(separated_res, separated_input)[0]
                    mse = mse_compute(separated_res, separated_input).cpu().item()
                    record.loc[len(record)] = [dc.batch_size, ac.num_bins, *list(map(len, res)), accuracy, ssim, psnr,
                                               mse, epsilon]
            record.to_json(r'evaluation/privacy parameters/epsilon plus.json', default_handler=str)
    elif mode == 'clipping bound':
        dataset = 'ImageNet'
        record = pd.DataFrame(columns=['batch size', 'bin num', 'res num', 'separated res', 'separated input',
                                       'overlapped res', 'overlapped input', 'meaningless', 'accuracy', 'ssim',
                                       'psnr', 'mse', 'clipping bound'])
        for bs in [16, 32, 64]:
            for rounds in range(5):
                fig_idx = torch.randint(0, 10000, (bs,))
                for cb in [100, 150, 200, 250, 300, 350, 400]:
                    dc = data_cfg_default(compress_image=True)
                    ac = attack_cfg_default()
                    dc.batch_size = bs
                    ac.num_bins = 1024
                    print('batch size: {0}, cb: {1}, round {2}'.format(dc.batch_size, cb, rounds))

                    res = proposed.perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, clipping_bound=cb,
                                                   dataset=dataset, use_dp=True, fig_idx=fig_idx)
                    accuracy = len(res[1]) / dc.batch_size
                    if len(res[2]) == 0:
                        # no reconstructed samples
                        continue

                    separated_input = torch.stack(res[2])
                    separated_input = separated_input * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

                    separated_res = torch.stack(res[1])
                    if dc.compress_image:
                        separated_res = compress_to_original(separated_res, separated_input.size())
                    separated_res = separated_res * dc.std.view(-1, 1, 1) + dc.mean.view(-1, 1, 1)

                    ssim = cw_ssim(separated_res, separated_input)[0]
                    psnr = psnr_compute(separated_res, separated_input)[0]
                    mse = mse_compute(separated_res, separated_input).cpu().item()
                    record.loc[len(record)] = [dc.batch_size, ac.num_bins, *list(map(len, res)), accuracy, ssim, psnr,
                                               mse, cb]
            record.to_json(r'evaluation/privacy parameters/clipping bound.json', default_handler=str)


# grad_comparison(mode='DP')
# value_comparison()
# factor_to_performance()
'''fig_quality_three_metrics(dataset='ImageNet')
fig_quality_three_metrics(dataset='CIFAR100')
fig_quality_three_metrics(dataset='Caltech256')
fig_quality_three_metrics(dataset='Flowers102')'''
privacy_setting(mode='epsilon')

