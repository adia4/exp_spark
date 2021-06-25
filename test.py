from pyspark.sql import SparkSession
import sparknlp
import json
import pandas as pd
import numpy as np
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
import nltk
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
# Available in PySpark >= 2.3.0 
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder\
	.appName("ner")\
	.master("spark://192.168.1.100:7077")\
	.config("spark.driver.memory","1g")\
	.config("spark.executor.memory","1g")\
	.config("spark.cores.max","1")\
	.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.1")\
	.config("spark.kryoserializer.buffer.max", "1000m")\
	.getOrCreate()
sc = spark.sparkContext
log_txt = sc.textFile("abc.txt")
header = log_txt.first()
MODEL_NAME = "onto_100"

class NLTKWordPunctTokenizer(
        Transformer, HasInputCol, HasOutputCol,
        DefaultParamsReadable, DefaultParamsWritable):

    stopwords = Param(Params._dummy(), "stopwords", "stopwords",
                      typeConverter=TypeConverters.toListString)


    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, stopwords=None):
        super(NLTKWordPunctTokenizer, self).__init__()
        self.stopwords = Param(self, "stopwords", "")
        self._setDefault(stopwords=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, stopwords=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setStopwords(self, value):
        return self._set(stopwords=list(value))

    def getStopwords(self):
        return self.getOrDefault(self.stopwords)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        stopwords = set(self.getStopwords())

        def f(s):
            tokens = nltk.tokenize.wordpunct_tokenize(s)
            # return [t for t in tokens if t.lower() not in stopwords]
            return " ".join([t for t in tokens if t.lower() not in stopwords])

        # t = ArrayType(StringType())
        t = StringType()
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))


documentAssembler = DocumentAssembler() \
    .setInputCol('words') \
    .setOutputCol('document')

# sentence = SentenceDetector().setInputCols('document').setOutputCol('sentence')

nltktokenizer = NLTKWordPunctTokenizer(
    inputCol="text", outputCol="words",  
    stopwords=nltk.corpus.stopwords.words('english'))

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

# ner_dl and onto_100 model are trained with glove_100d, so the embeddings in
# the pipeline should match
if (MODEL_NAME == "ner_dl") or (MODEL_NAME == "onto_100"):
    embeddings = WordEmbeddingsModel.pretrained('glove_100d') \
        .setInputCols(["document", 'token']) \
        .setOutputCol("embeddings")

# Bert model uses Bert embeddings
elif MODEL_NAME == "ner_dl_bert":
    embeddings = BertEmbeddings.pretrained(name='bert_base_cased', lang='en') \
        .setInputCols(['document', 'token']) \
        .setOutputCol('embeddings')

ner_model = NerDLModel.pretrained(MODEL_NAME, 'en') \
    .setInputCols(['document', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('ner_chunk')



nlp_pipeline = Pipeline(stages=[
    nltktokenizer,
    documentAssembler,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter
])


empty_df = spark.createDataFrame([['']]).toDF('text')
pipeline_model = nlp_pipeline.fit(empty_df)
df = spark.createDataFrame(pd.DataFrame({'text': [header]}))
result = pipeline_model.transform(df)

result.toPandas().to_csv('mycsv.csv')
