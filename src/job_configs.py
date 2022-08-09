import os
import copy
#import pretrained

JOB_CONFIGS = {}
JOB_CONFIGS['chris'] =  {
    'account_id':os.environ['EAI_ACCOUNT_ID'] ,
    'image': 'registry.console.elementai.com/snow.chris/fewshot_main',
    'data': [
        'snow.colab.public:/mnt/public',
        'snow.chris.home:/mnt/home'
    ],
    # NOTE: set WORLD_SIZE=0 if you want everything
    # in the main thread, otherwise multiprocessing
    # is used.
    #'environment_vars': ["WORLD_SIZE=0"],
    'restartable':True,
    'resources': {
        'cpu': 8,
        'mem': 16,
        'gpu_mem': 16,
        'gpu': 1,
        'gpu_model': '!A100'
    },
    'interactive': False,
    'bid': 9999,
}
# Add pretrained model env variables to this as well
#JOB_CONFIGS['chris']['environment_vars'] += \
#    [ "{}={}".format(k,v) for k,v in pretrained.models.items() ]

JOB_CONFIGS['chris_v2'] = copy.deepcopy(JOB_CONFIGS['chris'])
JOB_CONFIGS['chris_v2']['resources']['gpu'] = 1
JOB_CONFIGS['chris_v2']['environment_vars'] = [
    "DATASET_CLEVR_KIWI=/mnt/public/datasets/clevr-mrt/v2/train-val",
    "DATASET_CLEVR_KIWI_TEST=/mnt/public/datasets/clevr-mrt/v2/test",
    "DATASET_CLEVR_KIWI_META=/mnt/public/datasets/clevr-mrt/v2/metadata"
]

JOB_CONFIGS['chris_v1'] = copy.deepcopy(JOB_CONFIGS['chris'])
JOB_CONFIGS['chris_v1']['resources']['gpu'] = 1
JOB_CONFIGS['chris_v1']['environment_vars'] = [
    "DATASET_CLEVR_KIWI=/mnt/public/datasets/clevr-mrt/v1/train-val",
    "DATASET_CLEVR_KIWI_TEST=/mnt/public/datasets/clevr-mrt/v1/test",
    "DATASET_CLEVR_KIWI_META=/mnt/public/datasets/clevr-mrt/v1/metadata/chris-v4"
]