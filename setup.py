from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Pipeline Big Data appliqué à 298 000 mesures réglementaires d'autosurveillance (Agence Seine-Normandie, décembre 2022) pour répondre à deux problématiques de l'assainissement : (1) prédire la conformité réglementaire d'un rejet d'eaux usées traitées (classification binaire, 3 algorithmes comparés), (2) segmenter le parc des 1 533 STEU en profils de performance homogènes (clustering avec PCA et silhouette score).',
    author='Amine Harkane',
    license='MIT',
)
