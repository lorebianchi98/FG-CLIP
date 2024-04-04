from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset import PACCO2CLIPDataset
from src.eval_util import get_mean_rank_sum
from src.metrics import get_image_and_text_tensor, i2t, t2i
from src.loss import ContrastiveLoss
from src.plots_util import bcolors, plot_values, plot_losses

# from loss import ContrastiveLoss
from copy import deepcopy
import os
import matplotlib.pyplot as plt


# function to choice a negative image for each caption
def choice_negatives(annotations, images, metadata, strategy):
    # Random strategy
    # we couple each annotation with a random image (different from the correct one) from the batch to get negative samples
    if strategy == 'random':
        while True:
            perm = torch.randperm(images.shape[0])
            if not torch.all(metadata['image_id'][perm] == metadata['image_id']).item():
                return perm
            
    # Best score strategy
    # choosing as negative in the batch the element with higher dot product (different from the correct one)
    if strategy == 'best-score':
        mask = metadata['image_id'] == torch.diagonal(metadata['image_id'].expand(metadata['image_id'].shape[0], -1)).unsqueeze(1)
        scores = annotations @ images.transpose(1, 0) 
        scores = scores.masked_fill(mask.to(annotations.device), 0)
        perm = torch.argmax(scores, dim=1)
        return perm


def train(model, train_dataloader, contrastive_loss, optimizer, fgovd):
    # train the model for one epoch
    train_batch_losses = []
    device = next(model.parameters()).device
    
    for n_batch, batch in enumerate(tqdm(train_dataloader)):
        annotations = batch['annotation'].to(device)
        images = batch['image'].to(device)
        
        loss = contrastive_loss(images, annotations, fgovd=fgovd, return_similarity_mat=False)
        train_batch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.mean(torch.tensor(train_batch_losses)).item()

def trainBCE(model, train_dataloader, criterion, optimizer, negative_strategy):
    # train the model for one epoch
    train_batch_losses = []
    device = next(model.parameters()).device
    
    for n_batch, batch in enumerate(tqdm(train_dataloader)):
        annotations = batch['annotation'].to(device)
        images = batch['image'].to(device)
        metadata = batch['metadata']
        if negative_strategy == 'autodistillation':
            # targets = (F.normalize(annotations, p=2, dim=0) @ F.normalize(images, p=2, dim=0).T).float()
            # targets = (targets + 1) / 2
            targets = (F.normalize(images, p=2, dim=1) @ F.normalize(annotations, p=2, dim=1).T).float()
            out_score = model(images, annotations, ret_similarity_matrix=True)
        else:
            perm = choice_negatives(annotations, images, metadata, negative_strategy)
            # coupling annotations with negative images
            targets = torch.cat((torch.ones(images.shape[0]), torch.zeros(images.shape[0]))).to(device)
            images = torch.cat((images, images[perm]))
            annotations = torch.cat((annotations, annotations))
            
            out_score = model(images, annotations)
        
        loss = criterion(out_score, targets)
        train_batch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.mean(torch.tensor(train_batch_losses)).item()


def validate(model, val_dataloader, contrastive_loss, fgovd, verbose=True):
    # evaluate the model in the validation set
    device = next(model.parameters()).device
    val_batch_losses = []
    
    val_dataloader = tqdm(val_dataloader) if verbose else val_dataloader
    for n_batch, batch in enumerate(val_dataloader):
        annotations = batch['annotation'].to(device)
        images = batch['image'].to(device)
        
        with torch.no_grad():
            loss = contrastive_loss(images, annotations, fgovd)
    
        val_batch_losses.append(loss.item())
    return torch.mean(torch.tensor(val_batch_losses)).item()

def validateBCE(model, val_dataloader, criterion, negative_strategy):
    # evaluate the model in the validation set
    device = next(model.parameters()).device
    val_batch_losses = []
    
    for n_batch, batch in enumerate(tqdm(val_dataloader)):
        annotations = batch['annotation'].to(device)
        images = batch['image'].to(device)
        metadata = batch['metadata']
        if negative_strategy == 'autodistillation':
            # targets = (F.normalize(annotations, p=2, dim=0) @ F.normalize(images, p=2, dim=0).T).float()
            # targets = (targets + 1) / 2
            with torch.no_grad():
                # targets = (images @ annotations.T).float()
                targets = (F.normalize(images, p=2, dim=1) @ F.normalize(annotations, p=2, dim=1).T).float()
                out_score = model(images, annotations, ret_similarity_matrix=True)
        else:
            perm = choice_negatives(annotations, images, metadata, negative_strategy)
            targets = torch.cat((torch.ones(images.shape[0]), torch.zeros(images.shape[0]))).to(device)
            images = torch.cat((images, images[perm]))
            annotations = torch.cat((annotations, annotations))
            
            with torch.no_grad():
                out_score = model(images, annotations)
        
        loss = criterion(out_score, targets)
        val_batch_losses.append(loss.item())
    return torch.mean(torch.tensor(val_batch_losses)).item()
    

def do_train(model,
             train_dataset,
             val_dataset,
             train_config,
             seed=123,
             warmup=True,
             plot=False,
             loss_path=None,
             additional_val_dataset=None,
             results_path=None
             ):

    if type(train_dataset) != type(val_dataset):
        raise("Train and Validation datasets must be of the same type")    
    
    device = next(model.parameters()).device
    
    # mandatory parameters
    lr, ltype, num_epochs, batch_size = train_config['lr'], train_config['ltype'], train_config['num_epochs'], train_config['batch_size']
    # optional parameters
    margin = train_config.get('margin', 0.2)
    max_violation = train_config.get('max_violation', True)
    fgovd = train_config.get('fgovd', False)
    pacco_validation_sets_dir = train_config.get('pacco_validation_sets_dir', None) # if setted, it will be calculated the sum of the ranks on PACCO
    negative_strategy = train_config.get('negative_strategy', None)
    shuffle = train_config.get('shuffle', True)
    save_best_model = train_config.get('save_best_model', True)
    early_stopping = train_config.get('early_stopping', 0)
    retrieval_evaluation_set = train_config.get('retrieval_evaluation_set', None)
    
    num_epochs += 1 # in order to make epochs starting from 1 and having epoch 0 as warmup evaluation
    
    if isinstance(train_dataset, PACCO2CLIPDataset) and ltype != 'triplet':
        raise("FG-OVD training must have triplet loss")
    
    # setting manual seed
    torch.manual_seed(seed)
    
    # data loading
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if additional_val_dataset is not None:
        additional_batch_size = batch_size if isinstance(additional_val_dataset, PACCO2CLIPDataset) else batch_size // 4 # COCO must have lower bs
        additional_val_dataloader = DataLoader(additional_val_dataset, batch_size=additional_batch_size, shuffle=shuffle)

    # loss and optimizer
    if ltype == 'BCE':
        criterion = nn.BCELoss()
    elif ltype == 'MSE':
        criterion = nn.MSELoss()
    elif ltype == 'triplet':
        criterion = ContrastiveLoss(model, margin=margin, max_violation=max_violation, ltype=ltype)
        
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # if we want to perform evaluation of mean ranks on PACCO, we load the sets
    if pacco_validation_sets_dir is not None:
        datas = {}
        for filename in os.listdir(pacco_validation_sets_dir):
            datas[filename.split('.')[0]] = torch.load(os.path.join(pacco_validation_sets_dir, filename), map_location=device)
        print("Evaluating CLIP PACCO Mean Rank Sum...")
        mean_rank_sum = get_mean_rank_sum(None, datas, batch_size, device)
        clip_pacco_results = torch.ones(num_epochs) * mean_rank_sum
        pacco_results = torch.zeros(num_epochs)
        print(f"CLIP PACCO Mean Rank Sum={mean_rank_sum}")

    # if we want to evaluate retrieval performances, we load images and texts tensors and calculate original rsum
    if retrieval_evaluation_set is not None:
        print("Evaluating CLIP rsum...")
        retr_imm, retr_txt = get_image_and_text_tensor(retrieval_evaluation_set)
        r1, r5, r10, _, _ = t2i(retr_imm.numpy(), retr_txt.numpy())
        r1i, r5i, r10i, _, _ = i2t(retr_imm.numpy(), retr_txt.numpy())
        rsum = r1 + r5 + r10 + r1i + r5i + r10i
        clip_rsums = torch.ones(num_epochs) * rsum
        rsums = torch.zeros(num_epochs)
        print(f"CLIP rsum={rsum}")
          
    # losses declaration
    train_losses = torch.zeros(num_epochs)
    val_losses = torch.zeros(num_epochs)
    additional_val_losses = torch.zeros(num_epochs) if additional_val_dataset is not None else None
    
    
    # warmup evaluation
    if warmup:
        print("Warmup Evaluation...")
        start = 0
    else:
        start = 1
    
    count_no_improvements = 0
    for epoch in range(start, num_epochs):
        if epoch != 0:
            # Training step
            model.train()
            if ltype == 'BCE' or ltype == 'MSE':
                train_loss = trainBCE(model, train_dataloader, criterion, optimizer, negative_strategy)
            else:
                train_loss = train(model, train_dataloader, criterion, optimizer, fgovd)
            train_losses[epoch] = train_loss
        
        # Evaluate step
        model.eval()
        print("Performing Evaluation...")
        if ltype == 'BCE' or ltype == 'MSE':
            val_loss = validateBCE(model, val_dataloader, criterion, negative_strategy)
        else:
            val_loss = validate(model, val_dataloader, criterion, fgovd)
            
            # evaluating loss on eventual additional validation set
            if additional_val_dataset is not None:
                set_name = 'PACCO' if isinstance(additional_val_dataset, PACCO2CLIPDataset) else 'COCO'
                print(f"Evaluating loss on {set_name}...")
                additional_val_loss = validate(model, additional_val_dataloader, criterion, isinstance(additional_val_dataset, PACCO2CLIPDataset))
                additional_val_losses[epoch] = additional_val_loss
                print(f"Epoch {epoch}: {set_name} loss={additional_val_loss}")
            
            # evaluating mean rank sum on eventual PACCO validation sets
            if pacco_validation_sets_dir is not None:
                print("Evaluating PACCO Mean Rank Sum...")
                mean_rank_sum = get_mean_rank_sum(model, datas, batch_size, device)
                pacco_results[epoch] = mean_rank_sum
                print(f"Epoch {epoch}: PACCO_mean_rank_sum={mean_rank_sum}")
                
            # evaluating mean rank sum on eventual PACCO validation sets
            if retrieval_evaluation_set is not None:
                print("Evaluating rsum on COCO...")
                r1, r5, r10, _, _ = t2i(retr_imm.numpy(), retr_txt.numpy(), cross_attention=model)
                r1i, r5i, r10i, _, _ = i2t(retr_imm.numpy(), retr_txt.numpy(), cross_attention=model)
                rsum = r1 + r5 + r10 + r1i + r5i + r10i
                rsums[epoch] = rsum
                print(f"Epoch {epoch}: rsum on COCO={rsum}")
                
        val_losses[epoch] = val_loss
        print(f"Epoch {epoch}: train_loss={train_losses[epoch]} - val_loss={val_losses[epoch]}")
        
        if epoch == 0:
            continue
        
        # evaluating if best model and check early stopping
        if save_best_model and (epoch == 1 or val_losses[epoch] < min(val_losses[1:epoch]).item()):
            print(f"{bcolors.GREEN}Best validation loss, saving the model{bcolors.ENDC}")
            best_model = deepcopy(model)
            count_no_improvements = 0
        else:
            print(f"{bcolors.WARNING}No improvement in validation loss, model not saved{bcolors.ENDC}")
            count_no_improvements += 1
            if early_stopping > 0 and count_no_improvements >= early_stopping:
                print(f"{bcolors.WARNING}Reached {early_stopping} epochs without improvements, training ended{bcolors.ENDC}")
                train_losses = train_losses[:epoch + 1]
                val_losses = val_losses[:epoch + 1]
                break
    
    
    # Saving results and plotting graphs
    results = {}
    results['losses'] = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'additional_val_loss': additional_val_losses if additional_val_dataset is not None else None
    }
    results['mean_rank_sums'] = {
        'mean_rank_sums': pacco_results,
        'clip_mean_rank_sums': clip_pacco_results
    } if pacco_validation_sets_dir else None
    results['rsums'] = {
        'rsums': rsums,
        'clip_rsums': clip_rsums
    } if retrieval_evaluation_set is not None else None
    if results_path is not None:
        torch.save(results, results_path)
    
    if plot or loss_path:
        # defining labels for plots
        labels = ["Training Loss", "Validation Loss", 'Validation Loss']
        datasets = [train_dataset, val_dataset] if additional_val_losses is None else [train_dataset, val_dataset, additional_val_dataset]
        for i, dataset in enumerate(datasets):
            name = 'PACCO' if isinstance(dataset, PACCO2CLIPDataset) else 'COCO'
            labels[i] += f' {name}'
        
        
        plot_losses(train_losses, val_losses, additional_val_losses=additional_val_losses, labels=labels, plot=plot, path=loss_path, warmup=warmup)
        
        if plot:
            if pacco_validation_sets_dir is not None:
                plot_values([pacco_results, clip_pacco_results], ['Current Model', 'CLIP B/16'], 'Mean Rank Sum', 'Mean Rank sum on PACCO Validation set', warmup=warmup)
            
            if retrieval_evaluation_set is not None:
                plot_values([rsums, clip_rsums], ['Current Model', 'CLIP B/16'], 'rsum', 'rsum on COCO', warmup=warmup)
    
        
    model = best_model if save_best_model else model
    # return model, train_losses, val_losses
    return model

def get_name(num_heads, mlp_dim, num_attention_layers, dropout, batch_size, negative_strategy):
    name = f"{num_heads}-heads_{num_attention_layers}-atts_{batch_size}-bs_{negative_strategy}-negs"
    if mlp_dim:
        name += f"_{mlp_dim}-mlp"
    if dropout > 0:
        name += f"_{dropout}-dropout"
    name = name.replace('.', ',')
    return name

