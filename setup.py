from setuptools import setup

setup(name='align_rudder',
      version='0.0.1',
      install_requires=['gym', 'pandas', 'natsort', 'tqdm', 'biopython', 'scikit-learn',
                        'seaborn'],
      package_data={
          "align_rudder": ["demonstrations/**/*.npy", "demonstrations/**/*.p", "envs/policies/*.npy"]
      }
      )
