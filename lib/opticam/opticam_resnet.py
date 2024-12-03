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

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

import pdb

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

#def energy_point_game(bbox, saliency_map):
#    x1, y1, x2, y2 = bbox
#    w, h = saliency_map.shape
#
#    empty = torch.zeros((w, h))
#    empty[x1:x2, y1:y2] = 1
#    mask_bbox = saliency_map * empty
#
#    energy_bbox =  mask_bbox.sum()
#    energy_whole = saliency_map.sum()
#    
#    proportion = energy_bbox / energy_whole
#
#    return proportion

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
    labels = labels.to(device)
    masked_images = masked_images.to(device)

    logits = model(images).to(device)
    outputs = torch.nn.functional.softmax(logits, dim=1)
    if labels == None:
        predict_labels = outputs.argmax(axis=1)
    else:
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
    
    avg_gain = np.max((O-Y,np.zeros(Y.shape)),axis=0)/(1-Y)
    
    return avg_drop, avg_inc, avg_gain

def f_logit_predict(model, device, x, predict_labels):
    outputs = model(x).to(device)
    one_hot_labels = torch.eye(len(outputs[0])).to(device)[predict_labels]
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return j

def f_logit_predict_max(model, device, x, predict_labels):
    outputs = model(x).to(device)
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

def normlization_sigmoid(saliency_map):
    norm_saliency_map = 1/2*(nn.Tanh()(saliency_map)+1)
    return norm_saliency_map

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

class Basic_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
            max_iter=100,
            learning_rate=0.1,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min'
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
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
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images,alpha

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        output = self.fea_ext(images)
        if isinstance(labels, type(None)):
            labels = torch.argmax(output, dim=-1)
        else:
            labels = labels.to(self.device)

        feature = relu(self.fea_ext.activations[0]).to(self.device)
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step, alpha, feature
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images, self.max_iter, alpha, feature


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self


class Mimic_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
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
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
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


    def combine_activations(self, feature, conv, images, predict_labels):
        sm_tensor = conv(feature)
        # softmax
        sm_tensor = torch.nn.functional.softmax(sm_tensor, dim=1).to(self.device)
        # channel attention
        sm_tensor_v = sm_tensor.reshape((sm_tensor.shape[0],sm_tensor.shape[1],sm_tensor.shape[2]*sm_tensor.shape[3]))
        feature_v = feature.reshape((feature.shape[0],feature.shape[1],feature.shape[2]*feature.shape[3]))
        feature_v = feature_v.transpose(2,1)
        channel_map = torch.matmul(sm_tensor_v, feature_v)
        subset = channel_map[:,predict_labels]
        alpha = torch.cat([x[i:i+1] for i,x in enumerate(subset)],dim=0)
        alpha = alpha.reshape((alpha.shape[0],alpha.shape[1],1,1))
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        ## take the saliency map
        #subset = sm_tensor[:,predict_labels]
        #saliency_map = torch.cat([x[i:i+1] for i,x in enumerate(subset)],dim=0)
        #saliency_map = saliency_map.reshape((saliency_map.shape[0],1,saliency_map.shape[1],saliency_map.shape[2]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images, sm_tensor

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        output = self.fea_ext(images)
        pdb.set_trace()
        prob = nn.functional.softmax(output, dim=1).to(self.device)
        feature = relu(self.fea_ext.activations[0]).to(self.device)
        conv = torch.nn.Conv2d(feature.shape[1],output.shape[1],(1,1)).to(self.device)
        optimizer = optim.Adam(conv.parameters(), lr=self.learning_rate)
        prev = 1e10
        predict_labels = output.argmax(axis=1).to(self.device)
        f_images = self.get_f(images, predict_labels)
        mp = nn.MaxPool2d(feature.shape[2])
        L2 = torch.nn.MSELoss()
        cross_entropy = torch.nn.CrossEntropyLoss()

        for step in range(self.max_iter):
            norm_saliency_map, new_images, sm_tensor = self.combine_activations(feature, conv, images, predict_labels)
            mp_prob = mp(sm_tensor)
            loss_1 = cross_entropy(mp_prob.squeeze(), predict_labels)
            #loss_1 = L2(prob, mp_prob)
            loss_2 = self.get_loss(new_images, predict_labels, f_images)
            loss =  loss_2

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images, _ = self.combine_activations(feature, conv, images, predict_labels)
        return norm_saliency_map, new_images


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class Plus_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
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
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
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
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        sm = saliency_map
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images, sm, alpha

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        output = self.fea_ext(images)
        feature = relu(self.fea_ext.activations[0]).to(self.device)
        NormFea = torch.norm(feature, dim=(2,3))
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images, sm, alpha = self.combine_activations(feature, w, images)
            loss_cls = self.get_loss(new_images, predict_labels, f_images)
            loss_sparse = torch.norm(torch.exp(w),p=1)/feature.shape[1]
            dif = NormFea.to(self.device)*alpha.to(self.device).squeeze().squeeze()
            loss_context = dif.sum()#/feature.shape[1]
            featureN = feature/NormFea.unsqueeze(2).unsqueeze(3).repeat((1,1,feature.shape[2],feature.shape[3]))
            flag = (NormFea==0)
            featureN=torch.where(flag.unsqueeze(2).unsqueeze(3).repeat((1,1,feature.shape[2],feature.shape[3])),torch.zeros_like(featureN),featureN)
            sm = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*featureN).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
            sm = sm.reshape((sm.shape[0],sm.shape[2]*sm.shape[3]))
            loss_sim = torch.mm(sm,sm.transpose(1,0)).diag()
            loss = loss_cls + loss_sparse - loss_context - loss_sim.sum()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images,sm, alpha = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images, self.max_iter


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self


class TopK_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            topK=100
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.topK = topK

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
        #topK
        num, ind = torch.topk(w, self.topK, dim=1)
        # softmax
        num = torch.nn.functional.softmax(num, dim=1)
        alpha = torch.zeros_like(w).scatter(1, ind, num).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images,alpha

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
            norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step#, alpha, feature
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images, self.max_iter#, alpha, feature


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class Masks_InOut_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
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
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
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


    def combine_activations(self, feature, w, images, predict_labels, f_images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        out_images = (1- norm_saliency_map.repeat((1,images.shape[1],1,1)))* images

        nsm = norm_saliency_map
        pl = predict_labels
        fimg = f_images
        sort_nsm = nsm.view(-1).sort()[0]
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
        for th in threshold:
            th = sort_nsm[int(sort_nsm.shape[0]*th)]
            sm = (nsm>th).float()
            sm_img = sm.repeat((1,images.shape[1],1,1)) * images
            norm_saliency_map = torch.cat((norm_saliency_map, sm))
            new_images = torch.cat((new_images, sm_img))
            mask_out = (1- sm.repeat((1,images.shape[1],1,1)))* images
            out_images = torch.cat((out_images, mask_out))
            fimg = torch.cat((fimg, f_images))
            pl = torch.cat((pl, predict_labels))
        
        return norm_saliency_map, new_images,alpha, pl,fimg, out_images

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
            norm_saliency_map, new_images, alpha, pl, fimg, out_images = self.combine_activations(feature, w, images, predict_labels, f_images)
            loss_in = self.get_loss(new_images, pl, fimg)
            loss_out = self.get_loss(out_images, pl, fimg)
            loss = loss_in - loss_out
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step#, alpha, feature
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images, alpha, pl, fimg, out_images = self.combine_activations(feature, w, images, predict_labels, f_images)
        return norm_saliency_map, new_images, self.max_iter#, alpha, feature


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class Masks_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            threshold = [0.5,0.8,0.9,0.95,0.98],
            weight_lambda = 1
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.threshold = threshold
        self.weight_lambda=weight_lambda

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


    def combine_activations(self, feature, w, images, predict_labels, f_images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images

        nsm = norm_saliency_map
        pl = predict_labels
        fimg = f_images
        threshold = self.threshold
        for th in threshold:
            sm = (nsm>th).float()
            sm_img = sm.repeat((1,images.shape[1],1,1)) * images
            norm_saliency_map = torch.cat((norm_saliency_map, sm))
            new_images = torch.cat((new_images, sm_img))
            fimg = torch.cat((fimg, f_images))
            pl = torch.cat((pl, predict_labels))
        
        return norm_saliency_map, new_images,alpha, pl,fimg

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
        batch_size = images.shape[0]

        for step in range(self.max_iter):
            norm_saliency_map, new_images, alpha, pl, fimg = self.combine_activations(feature, w, images, predict_labels, f_images)
            loss1 = self.get_loss(new_images[0:batch_size], pl[0:batch_size], fimg[0:batch_size])
            loss2 = self.get_loss(new_images[batch_size:], pl[batch_size:], fimg[batch_size:])
            loss = loss1 + self.weight_lambda * loss2
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step#, alpha, feature
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images, alpha, pl, fimg = self.combine_activations(feature, w, images, predict_labels, f_images)
        return norm_saliency_map, new_images, self.max_iter#, alpha, feature


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class HalfB_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            threshold = 0.5,
            weight_lambda = 1
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.threshold = threshold
        self.weight_lambda=weight_lambda

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


    def combine_activations(self, feature, w, images, predict_labels, f_images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        sort_nsm = norm_saliency_map.view(-1).sort()[0]
        th = self.threshold
        th = sort_nsm[int(sort_nsm.shape[0]*th)]

        sm_b = (norm_saliency_map>th).float()
        sm = torch.clamp(norm_saliency_map, float(th.detach().cpu().numpy()),1)
        sm = sm*sm_b
        sm_img = sm.repeat((1,images.shape[1],1,1)) * images
        
        return sm, sm_img,alpha

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
        batch_size = images.shape[0]

        for step in range(self.max_iter):
            norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images, predict_labels, f_images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step#, alpha, feature
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images, predict_labels, f_images)
        return norm_saliency_map, new_images, self.max_iter#, alpha, feature


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.clamp(torch.sign(input),0,1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1,1)

class B_OptCAM:
    def __init__(self,
            model,
            device,
            target_layer,
            max_iter=100,
            learning_rate=0.01,
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            threshold = 0.5,
            weight_lambda = 1
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.threshold = threshold
        self.weight_lambda=weight_lambda

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


    def combine_activations(self, feature, w, images, predict_labels, f_images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images

        sort_nsm = norm_saliency_map.view(-1).sort()[0]
        th = self.threshold
        th = sort_nsm[int(sort_nsm.shape[0]*th)]

        sign = LBSign.apply
        sm = sign(norm_saliency_map - float(th.detach().cpu().numpy()))
        sm_img = sm.repeat((1,images.shape[1],1,1)) * images
        
        return sm, sm_img,alpha

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
        batch_size = images.shape[0]

        for step in range(self.max_iter):
            norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images, predict_labels, f_images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Attack stopped due to convergence...')
                    return norm_saliency_map, new_images, step#, alpha, feature
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images, alpha = self.combine_activations(feature, w, images, predict_labels, f_images)
        return norm_saliency_map, new_images, self.max_iter#, alpha, feature


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self
