from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml.feature import Bucketizer, OneHotEncoder, StringIndexer, VectorAssembler, PCA
from pyspark.ml.stat import Correlation, ChiSquareTest

# Initializing Spark session
spark =SparkSession.builder.appName("PredictiveMaintenance").getOrCreate()

# Loading the dataset
data = spark.read.csv("/content/ai4i2020.csv", header=True, inferSchema=True)

# Displaying columns names
data.printSchema()
print("Columns in the DataFrame:", data.columns)

# Checking for missing values
miss_val = data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])
miss_val.show()

# Handling missing values
num_columns = [field.name for field in data.schema.fields if field.dataType.typeName() in ['integer', 'double']]
for column_n in num_columns:
    mean_val = data.agg({column_n: "mean"}).first()[0]
    data = data.na.fill({column_n: mean_val})

# Binning continuous variable
splits = [-float("inf"), 300, 320, 340, float("inf")]
bucketizer = Bucketizer(splits=splits, inputCol="Air temperature [K]", outputCol="temp_binned")
data = bucketizer.transform(data)

column_name = "Machine failure"  
indexer = StringIndexer(inputCol=column_name, outputCol="machine_fail_index")

# Fitting and transforming data
data = indexer.fit(data).transform(data)

# One-Hot Encoding
encoder = OneHotEncoder(inputCols=["machine_fail_index"], outputCols=["machine_fail_ohe"])
data = encoder.fit(data).transform(data)

# Check for duplicate rows
dup = data.groupBy(data.columns).count().filter("count > 1")
dup.show()

# Correlation matrix
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=num_columns, outputCol=vector_col)
dataframe_vector = assembler.transform(data).select(vector_col)

correlation_matrix = Correlation.corr(dataframe_vector, vector_col).head()[0]
print(f"Pearson correlation matrix:\n{correlation_matrix}")

# Chi-Square test 
chi_sq_test = ChiSquareTest.test(data, featuresCol="machine_fail_ohe", labelCol="machine_fail_index")
chi_sq_test.show()

assembler = VectorAssembler(inputCols=num_columns, outputCol="features")
data_feat = assembler.transform(data)

# Applying PCA
pca = PCA(k=3, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(data_feat)
principal_comp_result = pca_model.transform(data_feat)

#Results
principal_comp_result.select("pca_features").show(truncate=False)

# Stop Spark session
spark.stop()
