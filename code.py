from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml.feature import Bucketizer, OneHotEncoder, StringIndexer, VectorAssembler, PCA
from pyspark.ml.stat import Correlation, ChiSquareTest

# Initialize Spark session
spark = SparkSession.builder.appName("PredictiveMaintenance").getOrCreate()

# Load the dataset
data = spark.read.csv("ai4i2020.csv", header=True, inferSchema=True)

# Display basic info and column names
data.printSchema()
print("Columns in the DataFrame:", data.columns)  # Debugging line

# Check for missing values
miss_val = data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])
miss_val.show()

# Handle missing values (example: fill with mean for numeric columns)
num_columns = [field.name for field in data.schema.fields if field.dataType.typeName() in ['integer', 'double']]
for column_n in num_columns:
    mean_val = data.agg({column_n: "mean"}).first()[0]
    data = data.na.fill({column_n: mean_val})

# Binning continuous variable (example: 'Air temperature [K]')
splits = [-float("inf"), 300, 320, 340, float("inf")]
bucketizer = Bucketizer(splits=splits, inputCol="Air temperature [K]", outputCol="temp_binned")
data = bucketizer.transform(data)

# Check the correct name for the categorical column
# Update 'Machine fail' based on the printed output
# Replace the column name below with the actual column name from the output
correct_column_name = "Machine failure"  # Change this if the printed column name is different
indexer = StringIndexer(inputCol=correct_column_name, outputCol="machine_fail_index")

# Fit and transform the data
data = indexer.fit(data).transform(data)

# One-Hot Encoding
encoder = OneHotEncoder(inputCols=["machine_fail_index"], outputCols=["machine_fail_ohe"])
data = encoder.fit(data).transform(data)

# Check for duplicate rows
dup = data.groupBy(data.columns).count().filter("count > 1")
dup.show()

# Correlation matrix (example: for numerical columns)
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=num_columns, outputCol=vector_col)
dataframe_vector = assembler.transform(data).select(vector_col)

correlation_matrix = Correlation.corr(dataframe_vector, vector_col).head()[0]
print(f"Pearson correlation matrix:\n{correlation_matrix}")

# Chi-Square test (for nominal features and the target variable)
chi_sq_test = ChiSquareTest.test(data, featuresCol="machine_fail_ohe", labelCol="machine_fail_index")
chi_sq_test.show()

# Prepare feature vector for PCA
assembler = VectorAssembler(inputCols=num_columns, outputCol="features")
data_feat = assembler.transform(data)

# Apply PCA
pca = PCA(k=3, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(data_feat)
principal_comp_result = pca_model.transform(data_feat)

# Show the result of PCA
principal_comp_result.select("pca_features").show(truncate=False)

# Stop Spark session
spark.stop()
