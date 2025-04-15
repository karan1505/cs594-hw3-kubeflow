import kfp
from pipeline import imdb_pipeline

kfp.compiler.Compiler().compile(imdb_pipeline, 'imdb_pipeline.yaml')