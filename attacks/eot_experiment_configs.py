import numpy as np


configs = [
{
    'dataset': 'indoor_scenes',
    'dataset_mode': 'test',
    'classif_model_name': 'resnet_indoor',
    'relight_model_name': 'multi_illumination_murmann',
    
    
    'relight_checkpoint_path': '../relighters/multi_illumination/'\
                               'checkpoints/relight/epoch_13.pth',
    
    # Relighter-specific constant.
    'relighter_eps': 1e-4,
    
    # Learning rate for attack gradient descent.
    'learning_rate': 0.05, 
    
    # Number of gradient descent iterations in the attack.
    'num_iterations': 5, 
    
    # Gamma correction constant for the multi_illumination relighter.
    'gamma': 1.3,
    
    # Radius of ball of inf-norm of allowed perturbations.
    'eps': np.linspace(0.002, 0.1, 3),
    
    # Constrain the perturbed image to be in the same class as
    # the original image.
    'attack_type': 'class_constrained_eot',
    
    # Target label is 9 (warehouse) for all non-warehouse images, otherwise 0 (airport)
    'targets': [9, 0],
    
    'debugging': False,
},
{
    'dataset': 'pubfig10',
    'dataset_mode': 'test',
    'classif_model_name': 'pubfig_facenet',
    'relight_model_name': 'multi_illumination_murmann',
    
    'relight_checkpoint_path': '../relighters/multi_illumination/'\
                               'checkpoints/relight/epoch_13.pth',
    'relighter_eps': 1e-4,
    'learning_rate': 0.05, 
    'num_iterations': 5, 
    'gamma': 1.3, 
    'eps': np.linspace(0.002, 0.1, 5),
    'attack_type': 'class_constrained_eot',
    'targets': 'TODO',
    'debugging': False,
},
{
    'dataset': 'pubfig10',
    'dataset_mode': 'test',
    'classif_model_name': 'pubfig_facenet',
    'relight_model_name': 'dpr',
    
    'relight_checkpoint_path': '../relighters/DPR/trained_model/trained_model_03.t7',
    'learning_rate': 0.05, 
    'num_iterations': 5, 
    'gamma': 1.3, 
    'eps': np.linspace(0.002, 0.1, 5),
    'attack_type': 'class_constrained_eot',
    'targets': 'TODO',
    'debugging': False,

}
]


def get_eot_experiment_configs():
    return configs