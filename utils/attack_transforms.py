import copy
import numpy as np
import os
import sys


sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..'))
from attacks import eot_attacks
from attacks import utils


class EOTAttackTransform(object):
    def __init__(self, attack_ratio, config):
        """ Initialize a transform that executes the EOT attack indicated by
        the config dictionary with attack_ratio probability (or, in rare cases 
        when an attack is not found, return a random perturbation) and returns
        the original image with 1 - attack_ratio probability.
        
        See attacks/eot_experiment_configs.py for example configs.
        
        Only the first epsilon value from the config will be used for the attack.
        """
        
        self.config = config
        self.attack_ratio = attack_ratio
        
        # Load classification model and relighter.
        self.classif_model = utils.load_classification_model(config['classif_model_name'], 
                                                             config['classif_mode'])
        self.relight_model = utils.load_relighting_model(config['relight_model_name'], 
                                                         config['relight_checkpoint_path'])
        
    
    def __call__(self, image, gt_label):

        # Execute attack.
        if np.random.random() < self.attack_ratio:
            config = self.config   # deepcopy ?

            
            config['eps'] = self.config['epses'][0]
            config['gt_label'] = gt_label

            # Set the target label for the attack on the current image.
            if gt_label == config['targets'][0]:
                config['target_label'] = config['targets'][1]
            else:
                config['target_label'] = config['targets'][0]
    
            result = eot_attacks.do_attack(self.relight_model, 
                                           self.classif_model, image, config)

            # Return the adversarial attack if one is found.
            if 'adv_img' in result:
                return result['adv_img']
            
            # Return an arbitrary perturbation if no attack was found.
            else:
                eps = self.config['epses'][0]
                perturbation = np.random.uniform(-eps, eps, size=image.shape)
                return image + perturbation
        
        # No attack, just the original image
        else:
            return image
    
