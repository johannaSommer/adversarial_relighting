import numpy as np
import torch

import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients

from .utils import (
    read_image,
    preprocess_relight_input,
    postprocess_relight_output,
    normalize_classifier_input,
    preprocess_classifier_input,
    load_relighting_model,
    load_classification_model,
)


def update_image_gd(crt_image, crt_grad, learning_rate):
    """
    crt_imgage.shape = [W, H, 3]
    """
    new_img = crt_image.clone().detach()
    new_img -= learning_rate * crt_grad.data.clone()
    new_img = new_img.clamp(0, 1)
    new_img = new_img.detach().cpu().numpy()
        
    return new_img


def get_classification_prediction(imgs, classif_model):
    classif_output = classif_model.forward(imgs)
    
    _, predicted = torch.max(classif_output, 1)
    
    return classif_output, predicted


def get_probs(classif_output):
    """Get the normalized classifier probabilities from the
    predicted scores.
    """
    return torch.nn.Softmax(dim=1)(classif_output).detach().cpu().numpy()
    

def do_eot_attack(relight_model, classif_model, img, config):
    """Perform an attack based on the Expectation Over Transformation (EOT)
    algorithm (Athalye, Anish, et al. "Synthesizing robust adversarial
    examples." 2017). 
    
    If the attack type is 'unconstrained_eot', then no additional 
    constraints are added. 
    
    If the attack type is 'class_constrained_eot', then we add an extra 
    constraint that forces the image we optimize, x', to also be labelled 
    with the same class as the original image x.
    """
    result = {}
    result['loss_hist'] = []
    result['adv_label'] = None
    result['relit_has_nan'] = False
    
    # Get the normalized classifier probabilities from the
    # predicted scores.
    # get_probs = lambda out: torch.nn.Softmax(dim=1)(out).detach().cpu().numpy()[0]
    
    # Prediction for the original image.
    classif_output, predicted = get_classification_prediction(
        preprocess_classifier_input(img.copy(), config),
        classif_model)
    result['orig_label'] = predicted[0].item()
    result['orig_probs'] = get_probs(classif_output)[0]
    
    # Use the classifier prediction as the ground truth if labels are
    # not available.
    if 'gt_label' not in config:
        config['gt_label'] = result['orig_label']
        
    # Do not perform an attack if the prediction is not correct.
    if result['orig_label'] != config['gt_label']:
        return result
        
    img_orig = img.copy()
    new_img = img.copy()
    for it in range(config['num_iterations']):
        try:
            new_img_tensor = torch.from_numpy(new_img).float().cuda()
        except: 
            new_img_tensor = torch.from_numpy(new_img).float()
        new_img_tensor.requires_grad = True

        # For the unconstrained_eot attack, we only need the classification
        # result for visualization purposes.
        if config['attack_type'] == 'unconstrained_eot':
            with torch.no_grad():
                classif_output, predicted = get_classification_prediction(
                    new_img_tensor.permute(2, 0, 1)[None, :],
                    classif_model
                )
        # For the class_constrained_eot attack, we want the prediction        
        # of the classification model for the root image, because we want
        # to add a loss term that will force the root image to be the same
        # class as the original image.
        elif config['attack_type'] == 'class_constrained_eot':
            classif_output, predicted = get_classification_prediction(
                new_img_tensor.permute(2, 0, 1)[None, :],
                classif_model
            )
                  
            y_target = torch.ones((1, )) * result['orig_label']
            try:
                y_target = y_target.long().cuda()
            except:
                y_target = y_target.long()
            
            loss_root_class = torch.nn.CrossEntropyLoss(reduction='mean')(classif_output, y_target)
            
        # Remember the root image and its label if it's the last iteration.
        if it == config['num_iterations'] - 1:
            result['root_img'] = new_img_tensor.detach().cpu().numpy()
            result['root_label'] = predicted[0].item()
            result['root_probs'] = get_probs(classif_output)[0]
        
        # Forward pass throught the relighting model.
        if config['relight_model_name'] == 'multi_illumination_murmann':
            sample, config['mean'] = preprocess_relight_input(new_img_tensor, config)
            pred_relight = relight_model(sample)
            pred_relight = postprocess_relight_output(pred_relight, config)
            config.pop('mean', None)
            
        elif config['relight_model_name'] == 'dpr':
            input_l, sh, config['input_ab'] = preprocess_relight_input(new_img_tensor, config)
            out_l, out_sh = relight_model(input_l, sh, 0)
            pred_relight = postprocess_relight_output(out_l, config)
            config.pop('input_ab', None)
        
        if torch.isnan(pred_relight).sum() > 0:
            result['relit_has_nan'] = True
            return result
        
        # Remember the adversarial image and the image difference
        # if this is the last iteration. Because the 8 relit images 
        # are very similar, we only keep the first one.
        if it == config['num_iterations'] - 1:
            result['adv_img'] = pred_relight[0].detach().cpu()
            result['adv_img'] = result['adv_img'].numpy().transpose((1, 2, 0))
            result['diff_img'] = result['adv_img'] - img
           
        # Get classification results for the relit images.
        pred_relight = normalize_classifier_input(pred_relight, config)
        classif_output, predicted = get_classification_prediction(
            pred_relight, 
            classif_model
        )
        
        # Remember the label of the adversarial image if this
        # is the last iteration. 
        if it == config['num_iterations'] - 1:
            result['adv_label'] = predicted[0].item()
            result['adv_probs'] = get_probs(classif_output)[0]

        # Construct the target label.
        y_target = torch.ones(pred_relight.shape[0])
        y_target *= config['target_label']
        try:
            y_target = y_target.long().cuda()
        except:
            y_target = y_target.long()
            
        # Backward pass through the entire pipeline.
        loss_relit_class = torch.nn.CrossEntropyLoss(reduction='mean')(classif_output, y_target)
        # loss_relit_class = torch.autograd.Variable(loss_relit_class, requires_grad = True)
        
        loss = loss_relit_class 
        if config['attack_type'] == 'class_constrained_eot':
            loss += loss_root_class
       
        result['loss_hist'].append(loss.item())
        zero_gradients(new_img_tensor)
        loss.backward()
        
        # Check if the gradients have NaN values in debugging mode.
        if config['debugging']:
            print('grad nan:', torch.isnan(new_img_tensor.grad.data).sum())

        # Do a gradient descent step on the image.
        new_img = update_image_gd(new_img_tensor,
                                  new_img_tensor.grad, 
                                  config['learning_rate'])
        
        # Project the new root image back on an L-inf ball of radius eps.
        new_img = np.maximum(np.minimum(new_img, img_orig + config['eps']), img_orig - config['eps'])
        
    return result


def do_random_root_attack(relight_model, classif_model, img, config):
    eps = config['eps']
    batch_size = config['batch_size']
    num_batches = config['num_batches']
    num_classes = config['num_classes']
    
    result = {}
    
    # Get the predictions on the original image.
    classif_output, predicted = get_classification_prediction(
        preprocess_classifier_input(img.copy(), config),
        classif_model)
    result['orig_label'] = predicted[0].item()
    result['orig_probs'] = get_probs(classif_output)[0]
    
    # Store prediction results on batch_size * num_batches images
    # in a neighbourhood of the original image (root images).
    predicted_root_counts = np.zeros(num_classes, dtype=int)
    predicted_root_avg_probs = np.zeros(num_classes, dtype=float)
    root_img = np.zeros_like(img)
    
    # Store prediction results on the relit variants of the root
    # images above. 
    predicted_adv_counts = np.zeros(num_classes, dtype=int)
    predicted_adv_avg_probs = np.zeros(num_classes, dtype=float)
    adv_img = np.zeros_like(img)
    
    img_noisy = np.zeros((batch_size, *img.shape))
    with torch.no_grad():
        for _ in range(num_batches):
            img_noisy[..., :] = img

            # noise = np.random.randn(*img_noisy.shape)
            # noise = np.maximum(np.minimum(noise, eps), -eps)
            
            noise = np.random.uniform(-eps, eps, size=img_noisy.shape)
            
            img_noisy += noise
            root_img += img_noisy.mean(axis=0)
            
            img_noisy_preprocessed = preprocess_classifier_input(img_noisy, config)
            classif_output, predicted = get_classification_prediction(
                img_noisy_preprocessed, classif_model)

            # Classify the current batch of noisy inputs.
            predicted = predicted.detach().cpu().numpy()
            predicted_counts = np.bincount(predicted, minlength=num_classes)
            predicted_root_counts += predicted_counts
            predicted_root_avg_probs += get_probs(classif_output).mean(axis=0)

            # Relight the current batch of noisy inputs. 
            img_noisy_tensor = torch.from_numpy(img_noisy).float().cuda()
            if config['relight_model_name'] == 'multi_illumination_murmann':
                sample, config['mean'] = preprocess_relight_input(
                    img_noisy_tensor, config)
                pred_relight = relight_model(sample)
                pred_relight = postprocess_relight_output(pred_relight, config)
                config.pop('mean', None)

            elif config['relight_model_name'] == 'dpr':
                input_l, sh, config['input_ab'] = preprocess_relight_input(
                    img_noisy_tensor, config)
                out_l, out_sh = relight_model(input_l, sh, 0)
                pred_relight = postprocess_relight_output(out_l, config)
                config.pop('input_ab', None)
            
            adv_img += pred_relight.mean(axis=0).detach().cpu().numpy().transpose((1, 2, 0))
            
            # Classify the batch of relit images. 
            pred_relight = normalize_classifier_input(pred_relight, config)
            classif_output, predicted = get_classification_prediction(
                pred_relight, 
                classif_model
            )
            
            predicted = predicted.detach().cpu().numpy()
            predicted_counts = np.bincount(predicted, minlength=num_classes)
            predicted_adv_counts += predicted_counts
            predicted_adv_avg_probs += get_probs(classif_output).mean(axis=0)
            
    predicted_root_avg_probs /= num_batches
    predicted_adv_avg_probs /= num_batches
    root_img /= num_batches
    adv_img /= num_batches
    
    result['root_img'] = root_img
    result['root_label'] = np.argmax(predicted_root_counts)
    result['root_probs'] = predicted_root_counts / predicted_root_counts.sum()

    result['adv_img'] = adv_img
    result['adv_label'] = np.argmax(predicted_adv_counts)
    result['adv_probs'] = predicted_adv_counts / predicted_adv_counts.sum()

    result['diff_img'] = result['adv_img'] - img
    
    return result


def do_attack(relight_model, classif_model, img, config):
    if config['attack_type'] == 'unrestricted_eot' or \
       config['attack_type'] == 'class_constrained_eot':
        result = do_eot_attack(
            relight_model, 
            classif_model, 
            img, 
            config
        )
    elif config['attack_type'] == 'random_root':
        result = do_random_root_attack(
            relight_model, 
            classif_model, 
            img, 
            config
        )
    else:
        raise NotImplementedError
    
    return result
