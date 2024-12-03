import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import sys
sys.path.append("/gpfsstore/rech/gmx/ucr34ay/pytorch-grad-cam/") 
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

import pdb

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_segmentation(path, dir_seg):
    name = os.path.basename(path).split('.')
    name_path = os.path.dirname(path).split('/')
    class_name = name_path[-1]
    mask_file_name = name[0] + '.png'
    path_mask = os.path.join(os.path.join(dir_seg,class_name), mask_file_name)
    mask_gt = np.array(Image.open(path_mask))
    mask_gt = mask_gt[:,:,1]*256 + mask_gt[:,:,0]
    mask_gt = torch.from_numpy(mask_gt.astype(np.float))
    return mask_gt


class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        #img /= 255.0

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        #img = img.transpose(1, 3).transpose(2, 3)
        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

def check_bounding_box(bbox, sm, img):
    b, c, w, h = img.shape

    empty = np.zeros((w, h))
    for box in bbox:
        empty[box.xslice,box.yslice]=1

    sm = torch.from_numpy(np.tile(sm,(1,c,1,1)))
    empty = torch.from_numpy(np.tile(empty,(1,c,1,1)))

    bbox_sm = sm * empty
    out_bbox_sm = sm * (1-empty)

    out_bbox_img = img * (1-empty)
    bbox_img = img * empty

    sm_img = sm * img
    out_bbox_sm_img = out_bbox_sm * img
    bbox_sm_img = bbox_sm * img

    return bbox_img, out_bbox_img, sm_img, out_bbox_sm_img,  bbox_sm_img

def energy_point_game(bbox, saliency_map):
    w, h = saliency_map.shape

    empty = np.zeros((w, h))
    for box in bbox:
        empty[box.xslice,box.yslice]=1
    mask_bbox = saliency_map * empty

    energy_bbox =  mask_bbox.sum()
    energy_whole = saliency_map.sum()
    
    proportion = energy_bbox / energy_whole

    return proportion

def averageDropIncrease(model, images, labels, masked_images, device):
    images = images.to(device)
    masked_images = masked_images.to(device)

    logits = model(images).to(device)
    outputs = torch.nn.functional.softmax(logits, dim=1)
    if labels==None:
        predict_labels = outputs.argmax(axis=1)
    else:
        labels = labels.to(device)
        predict_labels = labels
    logits_mask = model(masked_images).to(device)
    outputs_mask = torch.nn.functional.softmax(logits_mask, dim=1)

    one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
    Y = torch.masked_select(outputs, one_hot_labels.bool())
    Y = Y.data.cpu().detach().numpy()

    one_hot_labels_mask = torch.eye(len(outputs_mask[0]))[predict_labels].to(device)
    O = torch.masked_select(outputs_mask, one_hot_labels_mask.bool())
    O = O.data.cpu().detach().numpy()

    avg_drop = np.max((Y-O,np.zeros(Y.shape)),axis=0)/Y
    avg_inc =  np.greater(O,Y)
    #avg_drop = np.sum(np.max((Y-O,np.zeros(Y.shape)),axis=0)/Y)/O.shape[0]
    #avg_inc = np.sum(np.greater(O,Y))/O.shape[0]
    
    return avg_drop, avg_inc

def averageDropIncrease_f(model, images, labels, masked_images, device):
    images = images.to(device)
    masked_images = masked_images.to(device)
    shape = masked_images.shape
    #images = images.reshape((shape[0],shape[1],shape[2]*shape[3]))
    #images = images.transpose(1,2)
    masked_images = masked_images.reshape((shape[0],shape[1],shape[2]*shape[3]))
    masked_images = masked_images.transpose(1,2)

    logits = model(images,flag_model="part2").to(device)
    outputs = torch.nn.functional.softmax(logits, dim=1)
    if labels==None:
        predict_labels = outputs.argmax(axis=1)
    else:
        labels = labels.to(device)
        predict_labels = labels
    logits_mask = model(masked_images,flag_model="part2").to(device)
    outputs_mask = torch.nn.functional.softmax(logits_mask, dim=1)

    one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
    Y = torch.masked_select(outputs, one_hot_labels.bool())
    Y = Y.data.cpu().detach().numpy()

    one_hot_labels_mask = torch.eye(len(outputs_mask[0]))[predict_labels].to(device)
    O = torch.masked_select(outputs_mask, one_hot_labels_mask.bool())
    O = O.data.cpu().detach().numpy()

    avg_drop = np.max((Y-O,np.zeros(Y.shape)),axis=0)/Y
    avg_inc =  np.greater(O,Y)
    #avg_drop = np.sum(np.max((Y-O,np.zeros(Y.shape)),axis=0)/Y)/O.shape[0]
    #avg_inc = np.sum(np.greater(O,Y))/O.shape[0]
    
    return avg_drop, avg_inc

def f_logit_predict(model, device, x, predict_labels):
    outputs = model(x).to(device)
    one_hot_labels = torch.eye(len(outputs[0])).to(device)[predict_labels]
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return j

def f_logit_predict_f(model, device, x, predict_labels):
    shape = x.shape
    x = x.reshape((shape[0],shape[1],shape[2]*shape[3]))
    x = x.transpose(1,2)
    outputs = model(x,flag_model="part2").to(device)
    one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return j

def f_logit_predict_max(model, device, x, predict_labels):
    outputs = model(x).to(device)
    one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
    i, _ = torch.max((1-one_hot_labels).to(device)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return j - i

def f_logit_predict_max_f(model, device, x, predict_labels):
    shape = x.shape
    x = x.reshape((shape[0],shape[1],shape[2]*shape[3]))
    x = x.transpose(1,2)
    outputs = model(x, flag_model="part2").to(device)
    one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
    i, _ = torch.max((1-one_hot_labels).to(device)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return j - i

def f_cross_entropy(model, device, x, predict_labels):
    logits = model(x).to(device)
    loss = torch.nn.CrossEntropyLoss()
    output = loss(logits, predict_labels)
    return output

def normlization_max_min(saliency_map):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    min_value = saliency_map.view(saliency_map.size(0),-1).min(dim=-1)[0]
    delta = max_value - min_value
    min_value = min_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    delta = delta.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = (saliency_map - min_value) / delta
    return norm_saliency_map

def power_normalization(x, alpha):
    sign_term = torch.sign(x)
    abs_term = torch.abs(x)
    power_term = torch.pow(abs_term, alpha)
    norm_term = sign_term * power_term
    return norm_term

def normlization_max_min_power(saliency_map, power):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    min_value = saliency_map.view(saliency_map.size(0),-1).min(dim=-1)[0]
    delta = max_value - min_value 
    min_value = min_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    delta = delta.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = (saliency_map - min_value) / delta
    norm_saliency_map = torch.pow(norm_saliency_map, power)
    return norm_saliency_map

def normlization_sigmoid(saliency_map):
    norm_saliency_map = 1/2*(nn.Tanh()(saliency_map/2)+1)
    return norm_saliency_map

def normlization_softmax(saliency_map):
    shape = saliency_map.shape
    soft = nn.Softmax(dim=2)
    m_sm = soft(saliency_map.view(shape[0],shape[1],shape[2]*shape[3]))
    m_sm = m_sm.reshape(shape)
    return m_sm


def normlization_max(saliency_map):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    max_value = max_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = saliency_map / max_value
    return norm_saliency_map

def normlization_max_power(saliency_map, power):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    max_value = max_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = saliency_map / max_value
    norm_saliency_map = torch.pow(norm_saliency_map, power)
    return norm_saliency_map

def normlization_tanh(saliency_map):
    norm_saliency_map = nn.Tanh()(saliency_map)
    return norm_saliency_map

def get_cls_each_layer(attentions):
    width = int((attentions[0].shape[3]-1)**0.5)
    i = 0
    with torch.no_grad():
        for attention in attentions:
            feature = attention[:,:,0,1:]
            feature = feature.reshape(attention.shape[0],attention.shape[1],width,width)
            if i==0:
                cls_map = feature
            else:
                cls_map = torch.cat((cls_map,feature),axis=1)
    return cls_map

class Basic_OptCAM:
    def __init__(self,
            model,
            device,
            layer_name='attn_drop',
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_attention)
                
        self.attentions=[]

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())


    def get_f(self, x, y):
        return f_logit_predict(self.model, self.device, x, y)

    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        elif self.name_loss == 'plain':
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        else:
            raise Exception("Not Implemented")
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images


    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            output = self.model(images)
            feature = get_cls_each_layer(self.attentions).to(self.device)
        del self.attentions
        self.attentions = []
        torch.cuda.empty_cache()
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels 
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        with torch.no_grad():
            f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Optimization stopped due to convergence...')
                    return norm_saliency_map, new_images
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')
            del self.attentions
            self.attentions = []
            torch.cuda.empty_cache()
            del norm_saliency_map, new_images, loss
            torch.cuda.empty_cache()

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class OptCAM_A:
    def __init__(self,
            model,
            device,
            layer_name='attn_drop',
            name_attention=-1, 
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.target_layer = name_attention

        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_attention)
                
        self.attentions=[]

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())


    def get_f(self, x, y):
        return f_logit_predict(self.model, self.device, x, y)

    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        elif self.name_loss == 'plain':
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        else:
            raise Exception("Not Implemented")
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images


    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            output = self.model(images)
            feature = self.attentions[self.target_layer][:,:,:,1:].to(self.device)
            feature = feature.reshape((feature.shape[0],feature.shape[1]*feature.shape[2],feature.shape[3]))
            wh = int(np.sqrt(feature.shape[2]))
            feature = feature.reshape((feature.shape[0],feature.shape[1],wh,wh)).to(self.device)
        del self.attentions,wh
        self.attentions = []
        torch.cuda.empty_cache()
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels 
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        with torch.no_grad():
            f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Optimization stopped due to convergence...')
                    return norm_saliency_map, new_images
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')
            del self.attentions
            self.attentions = []
            torch.cuda.empty_cache()
            del norm_saliency_map, new_images, loss
            torch.cuda.empty_cache()

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class OptCAM_O:
    def __init__(self,
            model,
            device,
            target_layer,
            reshape_transform,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min'
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, reshape_transform)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm

    def get_f(self, x, y):
        if self.name_f =='logit_predict':
            return f_logit_predict(self.model, self.device, x, y)
        if self.name_f =='logit_predict_max':
            return f_logit_predict_max(self.model, self.device, x, y)
        if self.name_f == 'logit_vector':
            return self.model(x).to(self.device)
        if self.name_f == 'cross_entropy':
            return f_cross_entropy(self.model, self.device, x, y)
        else:
            raise Exception("Not Implemented")


    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        else:
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'softmax':
            return normlization_softmax(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        if self.name_norm == 'max_pow_2':
            return normlization_max_power(saliency_map, 2)
        if self.name_norm == 'max_pow_3':
            return normlization_max_power(saliency_map, 3)
        if self.name_norm == 'max_pow_4':
            return normlization_max_power(saliency_map, 4)
        if self.name_norm == 'tanh':
            return normlization_tanh(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        output = self.fea_ext(images)
        feature = relu(self.fea_ext.activations[0]).to(self.device)
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images, self.max_iter


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class OptCAM_I:
    def __init__(self,
            model,
            device,
            target_layer,
            reshape_transform,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min'
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, reshape_transform)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm

    def get_f(self, x, y):
        if self.name_f =='logit_predict':
            return f_logit_predict(self.model, self.device, x, y)
        if self.name_f =='logit_predict_max':
            return f_logit_predict_max(self.model, self.device, x, y)
        if self.name_f == 'logit_vector':
            return self.model(x).to(self.device)
        if self.name_f == 'cross_entropy':
            return f_cross_entropy(self.model, self.device, x, y)
        else:
            raise Exception("Not Implemented")


    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        else:
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        if self.name_norm == 'max_pow_2':
            return normlization_max_power(saliency_map, 2)
        if self.name_norm == 'max_pow_3':
            return normlization_max_power(saliency_map, 3)
        if self.name_norm == 'max_pow_4':
            return normlization_max_power(saliency_map, 4)
        if self.name_norm == 'tanh':
            return normlization_tanh(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        # upsampling
        saliency_map = F.interpolate(w,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        norm_saliency_map = norm_saliency_map.to(self.device)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        output = self.fea_ext(images)
        feature = relu(self.fea_ext.activations[0]).to(self.device)
        w = Variable(0.5*torch.ones((feature.shape[0],1,feature.shape[2],feature.shape[3]),dtype=torch.float), requires_grad=True)
        #w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images, self.max_iter


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class OptCAM_K:
    def __init__(self,
            model,
            device,
            layer_name='qkv',
            name_attention=-1, 
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.target_layer = name_attention

        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_attention)
                
        self.attentions=[]

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())


    def get_f(self, x, y):
        return f_logit_predict(self.model, self.device, x, y)

    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        elif self.name_loss == 'plain':
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        else:
            raise Exception("Not Implemented")
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images


    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            output = self.model(images)
            feature = self.attentions[self.target_layer].to(self.device)
            feature = feature.reshape(feature.shape[0],feature.shape[1],3,int(feature.shape[2]/3)).permute(2,0,3,1)
            _,feature,_ = feature.unbind(0)
            feature = feature[:,:,1:]
            wh = int(np.sqrt(feature.shape[2]))
            feature = feature.reshape((feature.shape[0],feature.shape[1],wh,wh)).to(self.device)
        del self.attentions,wh
        self.attentions = []
        torch.cuda.empty_cache()
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels 
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        with torch.no_grad():
            f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Optimization stopped due to convergence...')
                    return norm_saliency_map, new_images
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')
            del self.attentions
            self.attentions = []
            torch.cuda.empty_cache()
            del norm_saliency_map, new_images, loss
            torch.cuda.empty_cache()

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class OptCAM_Feature:
    def __init__(self,
            model,
            device,
            target_layer,
            reshape_transform,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min'
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm

    def get_f(self, x, y):
        if self.name_f =='logit_predict':
            return f_logit_predict_f(self.model[1], self.device, x, y)
        if self.name_f =='logit_predict_max':
            return f_logit_predict_max_f(self.model[1], self.device, x, y)
        #if self.name_f == 'logit_vector':
        #    return self.model(x).to(self.device)
        #if self.name_f == 'cross_entropy':
        #    return f_cross_entropy(self.model, self.device, x, y)
        else:
            raise Exception("Not Implemented")


    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        else:
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'softmax':
            return normlization_softmax(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        if self.name_norm == 'max_pow_2':
            return normlization_max_power(saliency_map, 2)
        if self.name_norm == 'max_pow_3':
            return normlization_max_power(saliency_map, 3)
        if self.name_norm == 'max_pow_4':
            return normlization_max_power(saliency_map, 4)
        if self.name_norm == 'tanh':
            return normlization_tanh(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_feature = norm_saliency_map.repeat((1,feature.shape[1],1,1)) * feature
        return norm_saliency_map, new_feature

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        feature = self.model[0](images)
        feature = self.model[1](feature,flag_model="part1")
        feature = reshape_transform(feature,14,14).to(self.device)
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        f_images = self.get_f(feature, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_feature = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_feature, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_feature, step
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_feature = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_feature, self.max_iter


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class OptCAM_CLS:
    def __init__(self,
            model,
            device,
            target_layer,
            reshape_transform,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'softmax'
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, reshape_transform)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm

    def get_f(self, x, y):
        if self.name_f =='logit_predict':
            return f_logit_predict(self.model, self.device, x, y)
        if self.name_f =='logit_predict_max':
            return f_logit_predict_max(self.model, self.device, x, y)
        if self.name_f == 'logit_vector':
            return self.model(x).to(self.device)
        if self.name_f == 'cross_entropy':
            return f_cross_entropy(self.model, self.device, x, y)
        else:
            raise Exception("Not Implemented")


    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        else:
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'softmax':
            return normlization_softmax(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        if self.name_norm == 'max_pow_2':
            return normlization_max_power(saliency_map, 2)
        if self.name_norm == 'max_pow_3':
            return normlization_max_power(saliency_map, 3)
        if self.name_norm == 'max_pow_4':
            return normlization_max_power(saliency_map, 4)
        if self.name_norm == 'tanh':
            return normlization_tanh(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        ## softmax
        #alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        alpha = w.to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        output = self.fea_ext(images)
        feature = relu(self.fea_ext.activations[0]).to(self.device)
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images, self.max_iter


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self
