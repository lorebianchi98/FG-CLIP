import numpy
from tqdm import tqdm
import torch
import json


def i2t(images, captions, npts=None, cross_attention=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if cross_attention:
        device = next(cross_attention.parameters()).device
    
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in tqdm(range(npts)):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if cross_attention is not None:
            ims_tensor = torch.tensor(im).to(device)
            queries_tensor = torch.tensor(captions).to(device)
            with torch.no_grad():
                d = cross_attention(ims_tensor.expand(queries_tensor.shape[0], -1), queries_tensor).cpu().detach().numpy()
        else:
            captions = captions.astype(numpy.float32)
            im = im.astype(numpy.float32)
            captions = captions / numpy.linalg.norm(captions, axis=0)
            im = im / numpy.linalg.norm(im, axis=0)
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, cross_attention=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if cross_attention:
        device = next(cross_attention.parameters()).device
    
    if npts is None:
        npts = images.shape[0] // 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in tqdm(range(npts)):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if cross_attention is not None:
            ims_tensor = torch.tensor(ims).to(device)
            queries_tensor = torch.tensor(queries).to(device)
            with torch.no_grad():
                d = numpy.array([cross_attention(ims_tensor, query.unsqueeze(0).expand(ims_tensor.shape[0], -1)).cpu().detach().numpy() for query in queries_tensor])
        else:
            queries = queries.astype(numpy.float32)
            ims = ims.astype(numpy.float32)
            queries = queries / numpy.linalg.norm(queries, axis=0)
            ims = ims / numpy.linalg.norm(ims, axis=0)
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def get_image_and_text_tensor(path):
    data = torch.load(path)
    images = {imm['id']: imm['imm_feats'] for imm in data['images']}
    
    annotations = {}
    for ann in data['annotations']:
        annotations[ann['image_id']] = [ann['ann_feats']] + annotations.get(ann['image_id'], [])
    
    
    imm_feats, ann_feats = None, None
    for imm_id in tqdm(annotations.keys()):
        if ann_feats is None:
            ann_feats = torch.stack(annotations[imm_id])
            imm_feats = images[imm_id].expand(len(annotations[imm_id]), -1)
        ann_feats = torch.cat((ann_feats, torch.stack(annotations[imm_id])))
        imm_feats = torch.cat((imm_feats, images[imm_id].expand(len(annotations[imm_id]), -1)))
    
    
    return imm_feats, ann_feats
    
    
def main():
    # from src.model import CrossAttentionModule
    
    # cross_attention = CrossAttentionModule(512, 64)
    # cross_attention.load_state_dict(torch.load('checkpoints/64heads-1attention-layer-rand.pth'))
    # cross_attention.to('cuda')
    # cross_attention.eval()
    
    images, texts = get_image_and_text_tensor('../features/ViT-B-16/val.json')
    print(f"Images {len(images)} -------- Texts {len(texts)}")
    print("CLIP results (t2i, i2t):")
    print(t2i(images.numpy(), texts.numpy(), cross_attention=None))
    print(i2t(images.numpy(), texts.numpy(), cross_attention=None))
    # print("CA results (t2i, i2t):")
    # print(t2i(images.numpy(), texts.numpy(), cross_attention=cross_attention))
    # print(i2t(images.numpy(), texts.numpy(), cross_attention=cross_attention))
    
if __name__ == '__main__':
    main()