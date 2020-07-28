import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..'))
from attacks import eot_attacks
from attacks import utils
from utils import labels_util


class AttackEvaluator(object):
    def __init__(self, config, results_csv_suffix):
        self.config = config
        self.results_csv_suffix = results_csv_suffix
        
        # Load classification model and relighter.
        self.classif_model = utils.load_classification_model(config['classif_model_name'], 
                                                             config['classif_mode'])
        self.relight_model = utils.load_relighting_model(config['relight_model_name'], 
                                                         config['relight_checkpoint_path'])
        
        # Load dataset object.
        self.dataset = utils.load_dataset(config['dataset'], config['dataset_mode'])
 
        # Load dataset labels.
        self.idx_to_label = labels_util.load_idx_to_label(config['dataset'])
        self.label_to_idx = {label : idx for idx, label in self.idx_to_label.items()}

    def check_valid_img(self, image):
        img_mean = image.mean()
        
        if img_mean > 0.95 or img_mean < 0.1:
            return False
        
        return True
        
    def evaluate(self):
        results = []
        success_rates = []
        untargeted_success_rates = []
        
        columns = ['eps', 'gt_label', 'orig_label', 'target_label', 'adv_label', \
                   'orig_prob', 'root_prob', 'adv_prob']

        visualize_every = 100
        visualize_counter = 0

        for eps in self.config['epses']:
            print('--------------------------------------------------------------------')
            print('eps: ', eps)
            
            self.config['eps'] = eps
            
            # How many times the classifier was correct on 
            # the original images.
            cnt_correct = 0

            # How many times (out of self.cnt_corrrect) the
            # attack was successful.
            cnt_adversarial = 0
            cnt_untargeted = 0
            
            for idx in range(len(self.dataset)):
                img, gt_label = self.dataset[idx]

                # This happens if the cropper does not detect an image
                if img is None:
                    continue  # We skip this image
        
                # Set the ground truth label, necessary for the attack.
                self.config['gt_label'] = gt_label
            
                # Set the target label for the attack on the current image.
                if gt_label == self.config['targets'][0]:
                    self.config['target_label'] = self.config['targets'][1]
                else:
                    self.config['target_label'] = self.config['targets'][0]
        
                crt_result = eot_attacks.do_attack(self.relight_model, 
                                                   self.classif_model, img, self.config)
            
                # Classification result was correct.
                if crt_result['orig_label'] == gt_label:
                    cnt_correct += 1
                    
                    # Attack was successful.
                    if crt_result['adv_label'] == self.config['target_label'] and \
                       crt_result['root_label'] == self.config['gt_label'] and \
                       self.check_valid_img(crt_result['adv_img']):
                        cnt_adversarial += 1

                    # Attack was a successful untargeted attack.
                    if crt_result['adv_label'] !=  crt_result['orig_label'] and \
                            crt_result['root_label'] == self.config['gt_label'] and \
                            self.check_valid_img(crt_result['adv_img']):
                        cnt_untargeted += 1
                        
                    # Log the results of the attack.
                    result_tuple = (eps, gt_label, crt_result['orig_label'], self.config['target_label'])
                    result_tuple += (crt_result['adv_label'], )
                    result_tuple += (crt_result['orig_probs'][gt_label], )
                    result_tuple += (crt_result['root_probs'][crt_result['root_label']], )
                    result_tuple += (crt_result['adv_probs'][crt_result['adv_label']], )
                    results.append(result_tuple)

                    if visualize_counter % visualize_every == 0:
                        utils.visualize_attack(img, crt_result, self.idx_to_label)
                    visualize_counter += 1
        
            success_rates.append(cnt_adversarial / cnt_correct)
            untargeted_success_rates.append(cnt_untargeted / cnt_correct)

            print('------------------------------------------------------------------------------')
            
        # Add nice plots
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(success_rates)), success_rates)
        ax.set_ylabel('Success rate')
        ax.set_xlabel('eps')
        ax.set_title('Attack success rates')
        ax.set_ylim([0,1])
        ax.set_xticks(np.arange(len(success_rates)))
        ax.set_xticklabels(list(map(lambda e : str(round(e, 3)), self.config['epses'])))
        plt.show()

        print(f"Targeted attack success rates: {success_rates}")
        print(f"Untargeted attack success rates: {untargeted_success_rates}")
        
        # Save results to csv
        csv_name = self.config['dataset'] + '_' + self.config['classif_model_name'] + '_'
        csv_name += self.config['relight_model_name'] + self.results_csv_suffix
        
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(csv_name, index=False)
