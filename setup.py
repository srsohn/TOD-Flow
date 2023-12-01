from setuptools import setup

setup(
        name='dialog_flow',
        version='1.0',
        description = 'dialog systems using graphs',
        #packages = ['dialog_flow'],
        install_requires=['numpy','ipython','matplotlib','absl-py','sentence_transformers','openai','tqdm','scipy','glob2','collection','joblib','pandas','scikit-learn','pyeda','torch','transformers','dataclasses','tiktoken']
        )
