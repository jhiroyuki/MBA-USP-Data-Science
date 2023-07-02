from pyspark.sql import SparkSession

# conectar ao cluster spark local
# Create a SparkSession
sc = SparkSession.builder \
    .appName("MyApp") \
    .master("local") \
    .getOrCreate()
#%%
sc.stop()