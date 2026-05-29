import os

DATA_DIR = os.environ.get('HAR_DATA_DIR', 'JPEGImages')

CLASSES = [
    'Applauding', 'Blowing_Bubbles', 'Brushing_Teeth', 'Cleaning_The_Floor',
    'Climbing', 'Cooking', 'Cutting_Trees', 'Cutting_Vegetables',
    'Drinking', 'Feeding_a_horse', 'Fishing', 'Fixing_a_bike',
    'Fixing_a_Car', 'Gardening', 'Holding_an_Umbrella', 'Jumping',
    'Looking_through_a_Microscope', 'Looking_through_a_Telescope',
    'Phoning', 'Playing_Guitar', 'Playing_Violin', 'Pouring_Liquid',
    'Pushing_a_Cart', 'Reading', 'Riding_a_Bike', 'Riding_a_Horse',
    'Rowing_a_Boat', 'Running', 'Shooting_an_Arrow', 'Smoking',
    'Taking_Photos', 'Texting_Message', 'Throwing_Frisby',
    'Using_a_Computer', 'Walking_the_dog', 'Washing_Dishes',
    'Watching_TV', 'Waving_Hands', 'Writing_on_a_Board', 'Writing_on_a_Book',
]

NUM_CLASSES = len(CLASSES)  # 40

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10
BATCH_SIZE = 32

# Hyperparameters used in the research paper experiments
MODEL_CONFIGS = {
    'vgg': {
        'lr': 0.00001,
        'weight_decay': 1e-4,
        'num_epochs': 10,
        'pretrained': True,
    },
    'resnet': {
        'lr': 0.0001,
        'weight_decay': 1e-4,
        'num_epochs': 10,
        'pretrained': True,
    },
    'densenet': {
        'lr': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 15,
        'pretrained': False,
        'dropout_prob': 0.5,
    },
    'googlenet': {
        'lr': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 15,
        'pretrained': False,
    },
    'pretrained_googlenet': {
        'lr': 0.0001,
        'weight_decay': 0.0,
        'num_epochs': 10,
        'pretrained': True,
    },
    'pretrained_densenet': {
        'lr': 0.0001,
        'weight_decay': 0.0,
        'num_epochs': 10,
        'pretrained': True,
    },
}
