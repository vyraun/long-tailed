# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from pdb import set_trace as bp

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('trigram_anti_focal_loss')
class TrigramAntiFocalLossCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, gamma):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.alpha = None
        self.gamma = gamma

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--gamma', type=int, metavar='N',
            help='Focal Loss Gamma',
        )

    def forward(self, model, sample, reduce=True, valid=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, valid=False):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        iterations = model.get_num_updates()
        base = 1000

        #if iterations < 10000:
        #    self.gamma = 0
        #elif iterations < 20000:
        #    self.gamma = 1
        #elif iterations < 30000:
        #    self.gamma = 2
        #elif iterations < 40000:
        #    self.gamma = 3
        #else:
        #    self.gamma = 4

        # gamma = max(1, np.log(iterations)/np.log(base))
        # self.gamma = gamma

        # Can keep gamma here, else just remove it

        #print(iterations, np.log(iterations)/np.log(base) )
        #print("Loss Iter = {} , Gamma Value = {}".format(iterations, torch.log(iterations))

        # Rescale Log Probabilities Based on Real Probabilities
        pt = Variable(lprobs.data.exp())
        pt = pt.view(pt.shape[0], -1, 4)
        pt = torch.sum(pt, dim=2)/4
        pt = torch.repeat_interleave(pt, 4, dim=1)

        # Use log(num_updates) with different bases

        if valid == False:
            #if iterations % 2 == 0:
                #print("FC Update") @ 1.2
            lprobs = (1.0 + pt)**self.gamma * lprobs

        #print(pt.shape, lprobs.shape)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
