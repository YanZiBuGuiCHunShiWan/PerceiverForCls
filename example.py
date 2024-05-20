import os
import time
import numpy as np
import torch.distributed as dist
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW,Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from autolab_core import YamlConfig
from dataset.customloader import get_dataloaders
from transformers import PerceiverModel, PerceiverConfig,PerceiverForSequenceClassification
from transformers.models.perceiver.modeling_perceiver import PerceiverAudioPreprocessor,PerceiverTextPreprocessor
from transformers.models.perceiver.modeling_perceiver import PerceiverClassificationDecoder
from typing import Union,Dict,List,Optional
from datetime import datetime

class CustomPerceiverTextPreprocessor(PerceiverTextPreprocessor):
    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__(config)
        '''支持接收预训练Embedding'''
        self.config = config
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
            
    def forward(self,pretrained_embeddings:Union[np.ndarray,torch.Tensor,None],pos=None,**kwargs):
        b,s,d = pretrained_embeddings.shape
        if not isinstance(pretrained_embeddings,torch.Tensor):
            pretrained_embeddings = torch.Tensor(pretrained_embeddings)
        assert d == self.config.d_model
        position_ids = torch.arange(0, s, device=pretrained_embeddings.device)
        embeddings = pretrained_embeddings + self.position_embeddings(position_ids)
        return embeddings,None,pretrained_embeddings
             
             
def get_model(model_config:YamlConfig):
    config = PerceiverConfig()
    config.d_latents = 256
    config.qk_channels = 768
    config.num_self_attends_per_block = 12
    config.attention_probs_dropout_prob = 0.3
    customtextpreprocessor = CustomPerceiverTextPreprocessor(config)
    model = PerceiverModel(
    config,
    input_preprocessor=customtextpreprocessor,
    decoder=PerceiverClassificationDecoder(
    config,
    num_channels=config.d_latents,
    trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
    use_query_residual=True))
    return model

def save_checkpoint(saved_path:str,model,
                    epoch,steps,optimizer,scheduler,metrics):
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
            'epoch':epoch,
            'classifier':model_to_save.state_dict(), #DDP
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'metrics': metrics
        },os.path.join(saved_path,"model_{epoch}_{steps}.pt".format(epoch=epoch,steps=steps)))
    logger.info("*********************Saving Model Paramters Succeed!********************")
    
def evaluate_performance(label_saved_path:str,model:PerceiverModel,dataloader:DataLoader,local_rank:int,writer:SummaryWriter,mode:str,modal:str,cretierion):
    temp_ground_truth = []
    temp_predict_labels = []
    eval_loss_list = []
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            batch_ground_truth = batch["level_gt"]
            inputs = {"input_ids":None,"attention_mask":None}
            labels = batch["level_gt"].long().cuda(local_rank,non_blocking=True)
            model_output = model(inputs)
            if hasattr(model_output,"logits"):
                loss = cretierion(model_output.logits,labels).to(local_rank)
                model_output = model_output.logits
            else:
                loss = cretierion(model_output,labels).to(local_rank)
            if mode=="test":
                eval_loss_list.append(loss.cpu().numpy())
            temp_ground_truth.extend(batch_ground_truth.numpy())
            model_predict = torch.argmax(model_output,dim=1).cpu().numpy()
            temp_predict_labels.extend(model_predict)
    
    temp_ground_truth = np.array(temp_ground_truth)
    temp_predict_labels = np.array(temp_predict_labels)
    precision,recall,f1,_ = precision_recall_fscore_support(temp_ground_truth,temp_predict_labels)
    accuracy = accuracy_score(temp_ground_truth,temp_predict_labels)
    logger.info("{} Metrics:\n class 0 precision: {:.2f}% class 1 precision: {:.2f}%.\n class 0 recall: {:.2f}% class 1 recall: {:.2f}%.\n class 0 f1: {:.2f}% class 1 f1: {:.2f}%.\n accuracy: {:.2f}% ".format(mode,precision[0]*100,precision[1]*100,
                                                                                                             recall[0]*100,recall[1]*100,
                                                                                                             f1[0]*100,f1[1]*100,
                                                                                                             accuracy*100))  
    return accuracy,precision,recall,f1    

def main():
    dist.init_process_group(backend="nccl")
    config_path = "config/perceiverIO/phq9_train.yaml"
    Train_config = YamlConfig(config_path)
    Traindataloader = None
    Testdataloader = None
    logger.info("数据集获取完毕")
    model= get_model(Train_config)
    now_str = datetime.now().strftime("%m-%d-%H:%M:%S")
    saved_path = os.path.join(Train_config["OUTPUT_DIR"],Train_config["SCALE"],now_str)
    logger_saved_path = os.path.join(Train_config["LOGGER_DIR"],Train_config["SCALE"],now_str)
    if not os.path.exists(logger_saved_path):
        os.makedirs(logger_saved_path)
    
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    Train_config.save(os.path.join(saved_path,"config.yaml"))
    writer = SummaryWriter(saved_path+'runs')
    logger.add(os.path.join(logger_saved_path,"roatated.log"), 
           format="{time} {level} {message}",
           rotation="7 days", 
           retention="10 days",
           compression="zip")
    ################ 把模型放到显卡上 ################
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank=int(os.environ.get('LOCAL_RANK'))
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)  # master gpu takes up extra memory
    torch.cuda.empty_cache()
    model.cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    cretierion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-5,weight_decay=2e-6)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)
    model.train()
    steps=0

    acc_buffer = []
    model_info_buffer = []
    STATE  = True
    less_avg_acc_times = 15
    temp_less_avg_acc_times = 0
    for epoch in range(Train_config["MODEL"]["EPOCHES"]):  # loop over the dataset multiple times
        if not STATE:
            logger.info("Training complete ! epoch:{epoch} steps:{steps}".format(epoch=epoch,steps=steps))
            break    
        for batch in tqdm(Traindataloader):
            # get the inputs; 


            labels = batch["level_gt"].long().cuda(local_rank,non_blocking=True)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            inputs = {"input_ids":None,"attention_mask":None}
            output= model(inputs=inputs)
            if hasattr(output,"logits"):
                loss = cretierion(output.logits,labels).to(local_rank)
            else:
                loss = cretierion(output,labels).to(local_rank)
            loss.backward()
            optimizer.step()

            # evaluate
            if rank==0:
                steps+=1
                if steps%10 == 0:
                    curren_lr = lr_scheduler.get_last_lr()[0]
                    logger.info("modal:{} current epoch:{}. current step:{}. lr :{:.5}. loss:{:.4f}".format(modal,
                        epoch,steps,curren_lr,loss.cpu().item()))
                    writer.add_scalar('Loss/train',loss.cpu().item(),global_step=steps)
                if steps%100 == 0:
                    model.eval()
                    logger.info("Start computing metrics on test datasets")
                    accuracy,precision,recall,f1=evaluate_performance(saved_path,model,Testdataloader,local_rank,writer,mode="test",modal=modal,cretierion=nn.CrossEntropyLoss())
                    writer.add_scalar("Accuracy/test",accuracy,global_step=steps)
                    metrics = {"f1":f1,"recall":recall,"precision":precision,"acc":accuracy}
                    if len(acc_buffer) < 3:
                        acc_buffer.append(accuracy)
                        model_info_buffer.append((epoch,steps))
                        save_checkpoint(saved_path,model,epoch,steps,optimizer,lr_scheduler,metrics)
                    else:
                        avg_acc = np.mean(acc_buffer)
                        if avg_acc < accuracy:
                           temp_less_avg_acc_times = 0
                           acc_buffer.append(accuracy)
                           model_info_buffer.append((epoch,steps))
                           acc_buffer.pop(0)
                           deleted_model_info = model_info_buffer.pop(0)
                           deleted_model_path = os.path.join(saved_path,"model_{epoch}_{steps}.pt".format(epoch=deleted_model_info[0],
                                                                                              steps=deleted_model_info[1]))

                           save_checkpoint(saved_path,model,epoch,steps,optimizer,lr_scheduler,metrics=metrics)
                           if os.path.exists(deleted_model_path):
                               os.remove(deleted_model_path)
                               logger.info("**************Deleted previous model.pt****************")
                           else:
                               logger.info("File {} does not exist.".format(deleted_model_path))
                        else:
                            temp_less_avg_acc_times += 1
                            if temp_less_avg_acc_times >= less_avg_acc_times:
                                STATE = False 
            model.train()
        
if __name__=="__main__":
    
    main()

    