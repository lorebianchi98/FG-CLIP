import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.model import CrossAttentionModule

# Code borrowed and adapted from https://github.com/mesnico/ALADIN/blob/master/alad/loss.py
class Contrastive(nn.Module):
    def __init__(self, sim=None, margin=0, max_violation=False, ltype='triplet'):
        super(Contrastive, self).__init__()
        self.margin = margin
        self.sim = sim
        self.max_violation = max_violation
        self.ltype = ltype
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_contrastive_loss(self, scores, fgovd=False):
        if self.ltype == 'infonce':
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * scores
            logits_per_text = logits_per_image.t()

            # compute bidirectional CE loss
            num_logits = logits_per_image.shape[0]
            labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)
            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2

        elif self.ltype == 'triplet':
            if not fgovd:
                diagonal = scores.diag().view(scores.size(0), 1)
                d1 = diagonal.expand_as(scores)
                d2 = diagonal.t().expand_as(scores)

                # compare every diagonal score to scores in its column
                # caption retrieval
                cost_s = (self.margin + scores - d1).clamp(min=0)
                # compare every diagonal score to scores in its row
                # image retrieval
                cost_im = (self.margin + scores - d2).clamp(min=0)

                # clear diagonals
                mask = torch.eye(scores.size(0)) > .5
                I = mask
                if torch.cuda.is_available():
                    I = I.to(scores.device)
                cost_s = cost_s.masked_fill_(I, 0)
                cost_im = cost_im.masked_fill_(I, 0)

                # keep the maximum violating negative for each query
                if self.max_violation:
                    cost_s = cost_s.max(1)[0]
                    cost_im = cost_im.max(0)[0]

                loss = cost_s.sum() + cost_im.sum()
            else:
                # if we are in the FG-OVD case, the score to maximize is always the first (positive caption)
                # in addition we perform only triplet loss keeping the image as anchor (row-wise optimization)
                positive_scores = scores[:, 0].view(scores.size(0), 1)
                d1 = positive_scores.expand_as(scores)

                # compare every diagonal score to scores in its column
                # caption retrieval
                cost_s = (self.margin + scores - d1).clamp(min=0)

                # mask with all elements True in the first column
                mask = torch.cat((torch.ones(scores.shape[0], 1), torch.zeros(scores.shape[0], scores.shape[1] - 1)), dim=1) > 0.5
                I = mask
                if torch.cuda.is_available():
                    I = I.to(scores.device)
                cost_s = cost_s.masked_fill_(I, 0)

                # keep the maximum violating negative for each query
                if self.max_violation:
                    cost_s = cost_s.max(1)[0]

                loss = cost_s.sum()

        else:
            raise ValueError(f'{self.ltype} not known!')
            
        return loss

class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, sim, margin=0, max_violation=False, ltype='triplet'):
        super(ContrastiveLoss, self).__init__(sim=sim, margin=margin, max_violation=max_violation, ltype=ltype)
        

    def forward(self, im, s, fgovd=False, return_similarity_mat=False):
        # compute image-sentence score matrix
        scores = self.sim(im, s, True, fgovd)
        loss = self.compute_contrastive_loss(scores, fgovd)
        if return_similarity_mat:
            return loss, scores
        else:
            return loss


def main():
    model = CrossAttentionModule(64,num_attention_layers=1) 
    loss = ContrastiveLoss(model)
    
if __name__ == '__main__':
    main()